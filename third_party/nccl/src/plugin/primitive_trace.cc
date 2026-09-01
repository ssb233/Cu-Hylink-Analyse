/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#ifdef NCCL_EXPERIMENT_PRIMITIVE_TRACE

#include "alloc.h"
#include "comm.h"
#include "device.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool ncclPrimitiveTracePath(struct ncclComm* comm, const char* base, char* path, size_t pathSize) {
  const char* rankMarker = strstr(base, "%r");
  int written;
  if (rankMarker == nullptr) {
    written = snprintf(path, pathSize, "%s.rank%d.jsonl", base, comm->rank);
  } else {
    int prefixLength = (int)(rankMarker - base);
    written = snprintf(path, pathSize, "%.*s%d%s", prefixLength, base, comm->rank, rankMarker + 2);
  }
  return written >= 0 && (size_t)written < pathSize;
}

void ncclPrimitiveTraceDump(struct ncclComm* comm) {
  struct ncclDevPrimitiveTrace* trace = comm->profiler.primitiveTrace;
  const char* base = getenv("NCCL_PRIMITIVE_TRACE_FILE");
  if (trace == nullptr || base == nullptr || *base == '\0') return;

  // The trace ring lives in device memory.  All NCCL streams have already been
  // synchronized by commDestroySync, so this is the only host export and is
  // outside every nccl-tests measured window.
  struct ncclDevPrimitiveTrace* hostTrace =
      static_cast<struct ncclDevPrimitiveTrace*>(calloc(1, sizeof(struct ncclDevPrimitiveTrace)));
  if (hostTrace == nullptr) {
    WARN("Could not allocate host staging memory for primitive trace on rank %d", comm->rank);
    return;
  }
  if (ncclCudaMemcpy(hostTrace, trace, 1) != ncclSuccess) {
    WARN("Could not copy device primitive trace to host for rank %d", comm->rank);
    free(hostTrace);
    return;
  }
  if (hostTrace->enabled == 0) {
    free(hostTrace);
    return;
  }

  char path[4096];
  if (!ncclPrimitiveTracePath(comm, base, path, sizeof(path))) {
    WARN("Primitive trace path is too long for rank %d", comm->rank);
    free(hostTrace);
    return;
  }

  FILE* file = fopen(path, "w");
  if (file == nullptr) {
    WARN("Could not open primitive trace file %s for rank %d", path, comm->rank);
    free(hostTrace);
    return;
  }

  fprintf(file,
          "{\"type\":\"meta\",\"rank\":%d,\"cudaDev\":%d,\"samplePeriod\":%u,"
          "\"storage\":\"device\",\"publishFence\":\"none\",\"maxSampledWorks\":%u,"
          "\"groups\":%d,\"roles\":%d,\"eventsPerRole\":%d,\"recordsPerChannel\":%d,"
          "\"slotsPerWork\":%d,\"recordBytes\":%zu}\n",
          comm->rank, comm->cudaDev, hostTrace->samplePeriod, hostTrace->maxSampledWorks,
          NCCL_PRIMITIVE_TRACE_GROUPS, NCCL_PRIMITIVE_TRACE_ROLES, NCCL_PRIMITIVE_TRACE_EVENTS_PER_ROLE,
          NCCL_PRIMITIVE_TRACE_RECORDS_PER_CHANNEL, NCCL_PRIMITIVE_TRACE_SLOTS_PER_WORK,
          sizeof(struct ncclDevPrimitiveTraceRecord));

  for (int channelId = 0; channelId < MAXCHANNELS; channelId++) {
    struct ncclDevPrimitiveTraceChannel* channel = &hostTrace->channels[channelId];
    for (int slot = 0; slot < NCCL_PRIMITIVE_TRACE_RECORDS_PER_CHANNEL; slot++) {
      const struct ncclDevPrimitiveTraceRecord* record = &channel->records[slot];
      if (record->valid == 0) continue;
      fprintf(file,
              "{\"type\":\"record\",\"channel\":%u,\"rank\":%u,\"group\":%u,"
              "\"role\":%u,\"primitive\":%u,\"phase\":%u,\"funcId\":%u,"
              "\"eventIndex\":%u,\"sequence\":%" PRIu64 ",\"step\":%" PRIu64 ",\"sliceBytes\":%" PRIu64 ","
              "\"startNs\":%" PRIu64 ",\"durationNs\":%" PRIu64 ",\"spinLoadCount\":%u,"
              "\"eventCount\":%u,\"slot\":%d}\n",
              record->channel, record->rank, record->group, record->role, record->primitive, record->phase,
              record->funcId, record->eventIndex, record->sequence, record->step, record->sliceBytes, record->startNs,
              record->durationNs, record->spinLoadCount, record->eventCount, slot);
    }
    if (channel->overflow != 0 || channel->droppedRecords != 0) {
      fprintf(file,
              "{\"type\":\"overflow\",\"channel\":%d,\"overflow\":%u,\"droppedRecords\":%u,"
              "\"dropReasons\":%u,\"dropDetail\":%u}\n",
              channelId, channel->overflow, channel->droppedRecords, channel->reserved[0], channel->reserved[1]);
    }
  }
  fclose(file);
  free(hostTrace);
  INFO(NCCL_INIT, "Primitive trace dumped to %s", path);
}

#endif
