/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/**
 * JNI wrapper for the lightweight GPU columnar format (.lenc):
 * FOR+BitPack, Dictionary+BitPack, RLE, lossless Scaled-Float FOR, and fused decode-in-register.
 */
public final class LencNative {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static final class ReadPlan implements AutoCloseable {
    private long handle;

    private ReadPlan(long handle) {
      this.handle = handle;
    }

    public long totalPackedBytes() {
      return planTotalBytesNative(handle);
    }

    public int numRows() {
      return planNumRowsNative(handle);
    }

    public void readToHost(String path, HostMemoryBuffer dst) {
      long bytes = totalPackedBytes();
      if (bytes > 0) {
        if (dst.getLength() < bytes) {
          throw new IllegalArgumentException(
              "Host buffer too small: " + dst.getLength() + " < " + bytes);
        }
        readToHostNative(path, handle, dst.getAddress());
      }
    }

    public Table decodeFromHost(HostMemoryBuffer src, byte[] dictAsDictionary32) {
      long addr = totalPackedBytes() > 0 ? src.getAddress() : 0L;
      return new Table(decodeFromHostNative(handle, addr, dictAsDictionary32));
    }

    public Table decodeAndEvalFused(HostMemoryBuffer src, int exprKind, long param0) {
      long addr = totalPackedBytes() > 0 ? src.getAddress() : 0L;
      return new Table(decodeAndEvalFusedNative(handle, addr, exprKind, param0));
    }

    @Override
    public void close() {
      if (handle != 0L) {
        long h = handle;
        handle = 0L;
        destroyPlanNative(h);
      }
    }
  }

  public static void writeTable(Table table, String path) {
    writeTableNative(table.getNativeView(), path);
  }

  public static ReadPlan planRead(String path, int[] colIndices) {
    return new ReadPlan(planReadNative(path, colIndices));
  }

  private static native void writeTableNative(long tableViewHandle, String path);
  private static native long planReadNative(String path, int[] colIndices);
  private static native long planTotalBytesNative(long planHandle);
  private static native int planNumRowsNative(long planHandle);
  private static native void readToHostNative(String path, long planHandle, long hostAddr);
  private static native long[] decodeFromHostNative(
      long planHandle, long hostAddr, byte[] dictAsDictionary32);
  private static native long[] decodeAndEvalFusedNative(
      long planHandle, long hostAddr, int exprKind, long param0);
  private static native void destroyPlanNative(long planHandle);
}
