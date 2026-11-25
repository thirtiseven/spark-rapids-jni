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

import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/**
 * JNI bridge for Tier 1 Standard Vortex GPU scanning & writing via official vortex-cuda-ffi
 * (Arrow C Device ABI -> cudf::from_arrow_device zero-copy GPU import).
 */
public final class VortexNative {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private VortexNative() {}

  public static void writeTable(Table table, String[] colNames, String path) {
    writeTableNative(table.getNativeView(), colNames, path);
  }

  public static final class DeviceScan implements AutoCloseable {
    private long handle;

    private DeviceScan(long handle) {
      this.handle = handle;
    }

    public Table nextBatch(byte[] dict32Mask) {
      if (handle == 0L) {
        return null;
      }
      long[] cols = nextBatchNative(handle, dict32Mask);
      return cols == null ? null : new Table(cols);
    }

    @Override
    public void close() {
      if (handle != 0L) {
        long h = handle;
        handle = 0L;
        closeScanNative(h);
      }
    }
  }

  public static DeviceScan openScan(String path, String[] colNames, long batchRows) {
    return new DeviceScan(openScanNative(path, colNames, batchRows));
  }

  private static native void writeTableNative(long tableView, String[] colNames, String path);

  private static native long openScanNative(String path, String[] colNames, long batchRows);

  private static native long[] nextBatchNative(long handle, byte[] dict32Mask);

  private static native void closeScanNative(long handle);
}
