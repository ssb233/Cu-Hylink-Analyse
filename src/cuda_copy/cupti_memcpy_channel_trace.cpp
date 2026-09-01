#include <cupti.h>
#include <cupti_activity.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>

#include <unistd.h>

namespace {

constexpr size_t kActivityBufferBytes = 32ULL * 1024ULL * 1024ULL;

class MemcpyTracer {
 public:
  void initialize() noexcept {
    const char* output = std::getenv("COPYBENCH_CUPTI_MEMCPY_OUTPUT");
    if (output == nullptr || output[0] == '\0') return;

    output_path_ = output;
    output_.open(output_path_, std::ios::out | std::ios::trunc);
    if (!output_) {
      error_ = "cannot open trace output: " + output_path_;
      writeError();
      return;
    }

    output_ << "pid,deviceId,contextId,streamId,channelID,channelType,"
               "copyKind,srcDeviceId,dstDeviceId,bytes,startNs,endNs,"
               "durationMs,correlationId,activityKind\n";
    output_.flush();

    const CUptiResult register_status = cuptiActivityRegisterCallbacks(
        &MemcpyTracer::requestBuffer, &MemcpyTracer::completeBuffer);
    if (register_status != CUPTI_SUCCESS) {
      error_ = "cuptiActivityRegisterCallbacks failed: " +
               std::to_string(static_cast<int>(register_status));
      writeError();
      return;
    }

    const CUptiResult memcpy_status =
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    const CUptiResult memcpy2_status =
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2);
    if (memcpy_status != CUPTI_SUCCESS || memcpy2_status != CUPTI_SUCCESS) {
      error_ = "cuptiActivityEnable failed: memcpy=" +
               std::to_string(static_cast<int>(memcpy_status)) +
               " memcpy2=" +
               std::to_string(static_cast<int>(memcpy2_status));
      writeError();
      return;
    }

    active_.store(true, std::memory_order_release);
  }

  void shutdown() noexcept {
    if (shutdown_started_.exchange(true, std::memory_order_acq_rel)) return;
    if (!active_.load(std::memory_order_acquire)) {
      if (output_.is_open()) writeMetadata(0, false);
      return;
    }

    const CUptiResult flush_status =
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    size_t dropped = 0;
    const CUptiResult dropped_status =
        cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
    if (flush_status != CUPTI_SUCCESS || dropped_status != CUPTI_SUCCESS) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (error_.empty()) {
        error_ = "CUPTI shutdown status: flush=" +
                 std::to_string(static_cast<int>(flush_status)) +
                 " dropped=" +
                 std::to_string(static_cast<int>(dropped_status));
      }
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      output_.flush();
    }
    writeMetadata(dropped, true);
  }

  static MemcpyTracer& instance() {
    static MemcpyTracer tracer;
    return tracer;
  }

 private:
  static void CUPTIAPI requestBuffer(uint8_t** buffer, size_t* size,
                                     size_t* max_num_records) {
    *size = kActivityBufferBytes;
    *max_num_records = 0;
    *buffer = static_cast<uint8_t*>(std::malloc(*size));
  }

  static void CUPTIAPI completeBuffer(CUcontext, uint32_t, uint8_t* buffer,
                                      size_t, size_t valid_size) {
    if (buffer == nullptr) return;
    MemcpyTracer& tracer = instance();
    CUpti_Activity* record = nullptr;
    CUptiResult status = CUPTI_SUCCESS;
    while ((status = cuptiActivityGetNextRecord(buffer, valid_size, &record)) ==
           CUPTI_SUCCESS) {
      tracer.record(record);
    }
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      std::lock_guard<std::mutex> lock(tracer.mutex_);
      if (tracer.error_.empty()) {
        tracer.error_ = "cuptiActivityGetNextRecord failed: " +
                         std::to_string(static_cast<int>(status));
      }
    }
    std::free(buffer);
  }

  void record(const CUpti_Activity* activity) noexcept {
    if (activity == nullptr) return;
    if (activity->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      const auto* row = reinterpret_cast<const CUpti_ActivityMemcpy5*>(activity);
      writeRow(row->kind, row->deviceId, row->contextId, row->streamId,
               row->channelID, row->channelType, row->copyKind, -1, -1,
               row->bytes, row->start, row->end, row->correlationId,
               activity->kind);
    } else if (activity->kind == CUPTI_ACTIVITY_KIND_MEMCPY2) {
      const auto* row =
          reinterpret_cast<const CUpti_ActivityMemcpyPtoP4*>(activity);
      writeRow(row->kind, row->deviceId, row->contextId, row->streamId,
               row->channelID, row->channelType, row->copyKind,
               static_cast<int>(row->srcDeviceId),
               static_cast<int>(row->dstDeviceId), row->bytes, row->start,
               row->end, row->correlationId, activity->kind);
    }
  }

  void writeRow(CUpti_ActivityKind activity_kind, uint32_t device_id,
                uint32_t context_id, uint32_t stream_id, uint32_t channel_id,
                CUpti_ChannelType channel_type, uint8_t copy_kind,
                int src_device_id, int dst_device_id, uint64_t bytes,
                uint64_t start, uint64_t end, uint32_t correlation_id,
                CUpti_ActivityKind record_kind) noexcept {
    const double duration_ms =
        end >= start ? static_cast<double>(end - start) / 1.0e6 : 0.0;
    std::lock_guard<std::mutex> lock(mutex_);
    output_ << static_cast<unsigned long long>(getpid()) << ','
            << device_id << ',' << context_id << ',' << stream_id << ','
            << channel_id << ',' << static_cast<int>(channel_type) << ','
            << static_cast<int>(copy_kind) << ',' << src_device_id << ','
            << dst_device_id << ',' << bytes << ',' << start << ',' << end
            << ',' << std::fixed << std::setprecision(6) << duration_ms << ','
            << correlation_id << ',' << static_cast<int>(record_kind) << '\n';
    ++row_count_;
  }

  void writeError() noexcept {
    if (output_path_.empty()) return;
    std::ofstream error_file(output_path_ + ".error", std::ios::out | std::ios::trunc);
    if (!error_file) return;
    error_file << error_ << '\n';
  }

  void writeMetadata(size_t dropped, bool initialized) noexcept {
    if (output_path_.empty()) return;
    std::ofstream metadata(output_path_ + ".meta.json",
                            std::ios::out | std::ios::trunc);
    if (!metadata) return;
    std::lock_guard<std::mutex> lock(mutex_);
    metadata << "{\n"
             << "  \"pid\": " << static_cast<unsigned long long>(getpid())
             << ",\n"
             << "  \"initialized\": " << (initialized ? "true" : "false")
             << ",\n"
             << "  \"rows\": " << row_count_ << ",\n"
             << "  \"droppedRecords\": " << dropped << "\n"
             << "}\n";
  }

  std::string output_path_;
  std::ofstream output_;
  std::mutex mutex_;
  std::string error_;
  std::atomic<bool> active_{false};
  std::atomic<bool> shutdown_started_{false};
  unsigned long long row_count_ = 0;
};

}  // namespace

__attribute__((constructor)) static void copybenchCuptiInitialize() {
  MemcpyTracer::instance().initialize();
}

__attribute__((destructor)) static void copybenchCuptiShutdown() {
  MemcpyTracer::instance().shutdown();
}
