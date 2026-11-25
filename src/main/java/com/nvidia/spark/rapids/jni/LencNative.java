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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.ast.CompiledExpression;

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

    public byte[] columnEncodings() {
      return planColumnEncodingsNative(handle);
    }

    public byte[] columnTypes() {
      return planColumnTypesNative(handle);
    }

    long nativeHandle() {
      return handle;
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

    public Table decodeFromHostSelected(
        HostMemoryBuffer src, byte[] dictAsDictionary32, ColumnView selectedRowIds) {
      long addr = totalPackedBytes() > 0 ? src.getAddress() : 0L;
      return new Table(decodeFromHostSelectedNative(
          handle, addr, dictAsDictionary32, selectedRowIds.getNativeView()));
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

  public static final class LtoProgram implements AutoCloseable {
    private long handle;

    private LtoProgram(long handle) {
      this.handle = handle;
    }

    public Table run(ReadPlan plan, HostMemoryBuffer hostSrc, String[][] dictLiteralsPerCol) {
      return runSelected(plan, hostSrc, dictLiteralsPerCol, null);
    }

    public Table runSelected(
        ReadPlan plan,
        HostMemoryBuffer hostSrc,
        String[][] dictLiteralsPerCol,
        ColumnView selectedRowIds) {
      long addr = plan.totalPackedBytes() > 0 ? hostSrc.getAddress() : 0L;
      long selView = selectedRowIds != null ? selectedRowIds.getNativeView() : 0L;
      return new Table(runLtoProgramNative(handle, plan.nativeHandle(), addr, dictLiteralsPerCol, selView));
    }

    @Override
    public void close() {
      if (handle != 0L) {
        long h = handle;
        handle = 0L;
        destroyLtoProgramNative(h);
      }
    }
  }

  public static LtoProgram compileLtoProgram(CompiledExpression... exprs) {
    long[] handles = new long[exprs.length];
    for (int i = 0; i < exprs.length; i++) {
      handles[i] = exprs[i].getJitNativeHandle();
    }
    return new LtoProgram(compileLtoProgramNative(handles));
  }

  public static ColumnVector compactMaskIndices(ColumnView boolMask) {
    return new ColumnVector(compactMaskIndicesNative(boolMask.getNativeView()));
  }

  public static ColumnVector decodeDictionary(ColumnView dictCol) {
    return new ColumnVector(decodeDictionaryNative(dictCol.getNativeView()));
  }

  public static ColumnVector remapDictionaryToCanonical(ColumnView dictCol, String[] queryLiterals) {
    return new ColumnVector(remapDictionaryToCanonicalNative(dictCol.getNativeView(), queryLiterals));
  }

  private static native void writeTableNative(long tableViewHandle, String path);
  private static native long planReadNative(String path, int[] colIndices);
  private static native long planTotalBytesNative(long planHandle);
  private static native int planNumRowsNative(long planHandle);
  private static native byte[] planColumnEncodingsNative(long planHandle);
  private static native byte[] planColumnTypesNative(long planHandle);
  private static native void readToHostNative(String path, long planHandle, long hostAddr);
  private static native long[] decodeFromHostNative(
      long planHandle, long hostAddr, byte[] dictAsDictionary32);
  private static native long[] decodeFromHostSelectedNative(
      long planHandle, long hostAddr, byte[] dictAsDictionary32, long selectedRowsView);
  private static native long[] decodeAndEvalFusedNative(
      long planHandle, long hostAddr, int exprKind, long param0);
  private static native void destroyPlanNative(long planHandle);
  private static native long decodeDictionaryNative(long colViewHandle);
  private static native long remapDictionaryToCanonicalNative(long colViewHandle, String[] literals);
  private static native long compactMaskIndicesNative(long maskViewHandle);
  private static native long compileLtoProgramNative(long[] compiledAstHandles);
  private static native long[] runLtoProgramNative(
      long ltoProgHandle, long planHandle, long hostAddr, String[][] dictLiteralsPerCol,
      long selectedRowsView);
  private static native void destroyLtoProgramNative(long ltoProgHandle);
}
