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

import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/**
 * JNI bridge for Tier 1 Standard Vortex GPU scanning & writing via official vortex-cuda-ffi
 * with single-pass zero-double-copy coalescing and GPU 2-Phase Late Materialization.
 */
public final class VortexNative {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private VortexNative() {}

  public static void writeTable(Table table, String[] colNames, String path) {
    writeTableNative(table.getNativeView(), colNames, path);
  }

  public static final class MultiFileScan implements AutoCloseable {
    private long handle;

    private MultiFileScan(long handle) {
      this.handle = handle;
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

  /** Opt V2: Scan a single Vortex file projecting colNames (optionally taking ONLY rowIndices!). */
  public static Table scanFileWithRows(
      String path, String[] colNames, long[] rowIndices, byte[] dict32Mask) {
    long[] cols = scanFileWithRowsNative(path, colNames, rowIndices, dict32Mask);
    return cols == null ? null : new Table(cols);
  }

  /** Opt V2: Compact a BOOL8 GPU mask column into strictly-sorted surviving row indices. */
  public static long[] maskToRowIndices(ColumnView maskCol) {
    return maskToRowIndicesNative(maskCol.getNativeView());
  }

  private static native void writeTableNative(long tableView, String[] colNames, String path);

  private static native long openMultiScanNative(
      String[] filePaths, String[] colNames, long batchRows, int prefetchCap);

  private static native long[] nextCoalescedBatchNative(
      long handle, byte[] dict32Mask, long coalesceTargetBytes);

  private static native void closeMultiScanNative(long handle);

  private static native long[] scanFileWithRowsNative(
      String path, String[] colNames, long[] rowIndices, byte[] dict32Mask);

  private static native long[] maskToRowIndicesNative(long maskColView);
}
