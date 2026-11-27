/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.MemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/**
 * JNI bridge for Tier 1+ Standard Vortex GPU scanning & writing via official vortex-cuda-ffi:
 *   - True Zero-Copy ArrowDeviceArray -> ColumnVector (zero GPU D2D copy, zero read-path sync!)
 *   - Single-pass zero-double-copy coalescing fallback
 *   - GPU 2-Phase Late Materialization via vx_cuda_scan_path_device_batches_with_rows
 */
public final class VortexNative {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private VortexNative() {}

  public static void writeTable(Table table, String[] colNames, String path) {
    writeTableNative(table.getNativeView(), colNames, path);
  }

  private static final class ArrowOwnerCleaner extends MemoryBuffer.MemoryBufferCleaner {
    private long ownerHandle;

    ArrowOwnerCleaner(long ownerHandle) {
      this.ownerHandle = ownerHandle;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      if (ownerHandle != 0L) {
        long h = ownerHandle;
        ownerHandle = 0L;
        releaseArrowOwnerNative(h);
        return true;
      }
      return false;
    }

    @Override
    public boolean isClean() {
      return ownerHandle == 0L;
    }
  }

  public static final class MultiFileScan implements AutoCloseable {
    private long handle;

    private MultiFileScan(long handle) {
      this.handle = handle;
    }

    /**
     * True Zero-Copy ArrowDeviceArray -> ColumnVector[]:
     * Zero GPU D2D copy and zero read-path stream.sync(). All columns share one ref-counted
     * DeviceMemoryBuffer whose cleaner releases the ArrowDeviceArray when the last column closes.
     */
    public ColumnVector[] nextZeroCopyBatch(byte[] dict32Mask) {
      if (handle == 0L) {
        return null;
      }
      long[] res = nextZeroCopyBatchNative(handle, dict32Mask);
      if (res == null) {
        return null;
      }
      long ownerPtr = res[0];
      long approxBytes = res[1];
      int ncols = res.length - 2;
      ColumnVector[] cvs = new ColumnVector[ncols];
      boolean ok = false;
      DeviceMemoryBuffer ownerBuf = DeviceMemoryBuffer.fromExternalCleaner(
          0L, approxBytes, new ArrowOwnerCleaner(ownerPtr));
      try {
        for (int i = 0; i < ncols; i++) {
          cvs[i] = ColumnVector.fromViewWithContiguousAllocation(res[2 + i], ownerBuf);
        }
        ok = true;
        return cvs;
      } finally {
        ownerBuf.close();
        if (!ok) {
          for (ColumnVector cv : cvs) {
            if (cv != null) {
              cv.close();
            }
          }
        }
      }
    }

    public static final class BatchedTable {
      public final ColumnVector[] columns;
      public final int[] fileIndices;
      public final int[] rowCounts;

      BatchedTable(ColumnVector[] columns, int[] fileIndices, int[] rowCounts) {
        this.columns = columns;
        this.fileIndices = fileIndices;
        this.rowCounts = rowCounts;
      }
    }

    public BatchedTable nextBatchedMultiFile(int maxBatches, long maxRows, long targetBytes) {
      if (handle == 0L) {
        return null;
      }
      long[] packed = nextBatchedMultiFileNative(handle, maxBatches, maxRows, targetBytes);
      if (packed == null) {
        return null;
      }
      long ownerPtr = packed[0];
      long approxBytes = packed[1];
      int nc = (int) packed[2];
      int nRuns = (int) packed[3 + nc];
      int[] fIdx = new int[nRuns];
      int[] rCnt = new int[nRuns];
      for (int i = 0; i < nRuns; i++) {
        fIdx[i] = (int) packed[4 + nc + 2 * i];
        rCnt[i] = (int) packed[4 + nc + 2 * i + 1];
      }
      ColumnVector[] cvs = new ColumnVector[nc];
      boolean ok = false;
      if (ownerPtr != 0L) {
        DeviceMemoryBuffer ownerBuf = DeviceMemoryBuffer.fromExternalCleaner(
            0L, approxBytes, new ArrowOwnerCleaner(ownerPtr));
        try {
          for (int i = 0; i < nc; i++) {
            cvs[i] = ColumnVector.fromViewWithContiguousAllocation(packed[3 + i], ownerBuf);
          }
          ok = true;
          return new BatchedTable(cvs, fIdx, rCnt);
        } finally {
          ownerBuf.close();
          if (!ok) {
            for (ColumnVector cv : cvs) {
              if (cv != null) {
                cv.close();
              }
            }
          }
        }
      } else {
        try {
          for (int i = 0; i < nc; i++) {
            cvs[i] = new ColumnVector(packed[3 + i]);
          }
          ok = true;
          return new BatchedTable(cvs, fIdx, rCnt);
        } finally {
          if (!ok) {
            for (ColumnVector cv : cvs) {
              if (cv != null) {
                cv.close();
              }
            }
          }
        }
      }
    }

    public int getLastFileIndex() {
      if (handle == 0L) {
        return 0;
      }
      return getLastFileIndexNative(handle);
    }

    public Table nextCoalescedBatch(byte[] dict32Mask, long coalesceTargetBytes) {
      if (handle == 0L) {
        return null;
      }
      long[] cols = nextCoalescedBatchNative(handle, dict32Mask, coalesceTargetBytes);
      return cols == null ? null : new Table(cols);
    }

    @Override
    public void close() {
      if (handle != 0L) {
        long h = handle;
        handle = 0L;
        closeMultiScanNative(h);
      }
    }
  }

  public static MultiFileScan openMultiScan(
      String[] filePaths, String[] colNames, long batchRows, int prefetchCap) {
    return new MultiFileScan(openMultiScanNative(filePaths, colNames, batchRows, prefetchCap));
  }

  public static Table scanFileWithRows(
      String path, String[] colNames, long[] rowIndices, byte[] dict32Mask) {
    long[] cols = scanFileWithRowsNative(path, colNames, rowIndices, dict32Mask);
    return cols == null ? null : new Table(cols);
  }

  public static long[] maskToRowIndices(ColumnView maskCol) {
    return maskToRowIndicesNative(maskCol.getNativeView());
  }

  /**
   * Streams multi-file Vortex compressed segments (FSST / FastLanes BitPacked+FoR / ALP / Dict)
   * directly into a libcudf WholeStageNative.HashAggTask WITHOUT materializing uncompressed HBM columns!
   */
  public static long streamCompressedPathsToWholeStageAgg(
      String[] paths,
      String[] colNames,
      WholeStageNative.HashAggTask aggTask,
      long[] probeHandles,
      ColumnView[] dimCols) {
    long[] dViews = new long[dimCols.length];
    for (int i = 0; i < dimCols.length; i++) {
      dViews[i] = dimCols[i].getNativeView();
    }
    return streamCompressedPathsToWholeStageAggNative(
        paths, colNames, aggTask.getNativeHandle(), probeHandles, dViews);
  }

  private static native void writeTableNative(long tableView, String[] colNames, String path);

  private static native long openMultiScanNative(
      String[] filePaths, String[] colNames, long batchRows, int prefetchCap);

  private static native long[] nextZeroCopyBatchNative(long handle, byte[] dict32Mask);

  private static native long[] nextBatchedMultiFileNative(
      long handle, int maxBatches, long maxRows, long targetBytes);

  private static native int getLastFileIndexNative(long handle);

  private static native void releaseArrowOwnerNative(long ownerPtr);

  private static native long[] nextCoalescedBatchNative(
      long handle, byte[] dict32Mask, long coalesceTargetBytes);

  private static native void closeMultiScanNative(long handle);

  private static native long[] scanFileWithRowsNative(
      String path, String[] colNames, long[] rowIndices, byte[] dict32Mask);

  private static native long[] maskToRowIndicesNative(long maskColView);

  private static native long streamCompressedPathsToWholeStageAggNative(
      String[] paths, String[] colNames, long aggTaskHandle,
      long[] probeHandles, long[] dimColViews);
}
