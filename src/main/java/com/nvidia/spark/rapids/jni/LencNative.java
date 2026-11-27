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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/**
 * Shared JNI entry points for GPU-Native Lightweight Shuffle (internal Spark Exchange codec),
 * Generic Compiler-Level NVRTC Table Reduce / GroupBy, and DICTIONARY32 helpers.
 * (External columnar file scanning uses Parquet or Standard Vortex via {@link VortexNative}.)
 */
public final class LencNative {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private LencNative() {}

  public static ColumnVector decodeDictionary(ColumnView dictCol) {
    return new ColumnVector(decodeDictionaryNative(dictCol.getNativeView()));
  }

  public static ColumnVector remapDictionaryToCanonical(ColumnView dictCol, String[] literals) {
    return new ColumnVector(remapDictionaryToCanonicalNative(dictCol.getNativeView(), literals));
  }

  public static final class JitReduceProgram implements AutoCloseable {
    private long handle;

    private JitReduceProgram(long handle) {
      this.handle = handle;
    }

    public Table runOnTable(
        Table table,
        String[][] dictLits,
        int[] groupKeySlots,
        int maxGroups,
        byte[] aggKinds) {
      return new Table(runJitReduceTableNative(
          handle, table.getNativeView(), dictLits, groupKeySlots, maxGroups, aggKinds));
    }

    public Table runHighCardGroupBy(
        Table groupKeys,
        Table valueCols,
        String[][] dictLits,
        byte[] aggKinds) {
      return new Table(runJitReduceHighCardNative(
          handle, groupKeys.getNativeView(), valueCols.getNativeView(), dictLits, aggKinds));
    }

    @Override
    public void close() {
      if (handle != 0L) {
        long h = handle;
        handle = 0L;
        destroyJitReduceNative(h);
      }
    }
  }

  public static JitReduceProgram compileJitReduce(String cudaSource, String kernelName) {
    return new JitReduceProgram(compileJitReduceNative(cudaSource, kernelName));
  }

  public static DeviceMemoryBuffer[] splitAndEncodeShuffleToDevice(Table table, int... splits) {
    DeviceMemoryBuffer[] ret = new DeviceMemoryBuffer[2];
    boolean ok = false;
    try {
      long[] v = splitAndEncodeShuffleNative(table.getNativeView(), splits);
      ret[0] = DeviceMemoryBuffer.fromRmm(v[0], v[1], v[2]);
      ret[1] = DeviceMemoryBuffer.fromRmm(v[3], v[4], v[5]);
      ok = true;
      return ret;
    } finally {
      if (!ok) {
        if (ret[0] != null) ret[0].close();
        if (ret[1] != null) ret[1].close();
      }
    }
  }

  public static Table assembleAndDecodeShuffleFromDevice(
      HostMemoryBuffer dataHost,
      DeviceMemoryBuffer dataDev,
      long[] hostOffsets,
      DType[] dtypes,
      byte[] dictAsDictionary32) {
    int[] typeIds = new int[dtypes.length];
    int[] scales = new int[dtypes.length];
    for (int i = 0; i < dtypes.length; i++) {
      typeIds[i] = dtypes[i].getTypeId().getNativeId();
      scales[i] = dtypes[i].getScale();
    }
    return new Table(assembleAndDecodeShuffleNative(
        dataHost.getAddress(), dataDev.getAddress(), dataDev.getLength(),
        hostOffsets, typeIds, scales, dictAsDictionary32));
  }

  /** Column-pruned GPU decode for LightweightCachedBatchSerializer (only H2D-copies selected cols!). */
  public static Table decodeCachedLshfProjected(
      byte[] lshfBytes, int[] selectedCols, DType[] selectedDTypes) {
    int[] typeIds = new int[selectedDTypes.length];
    int[] scales = new int[selectedDTypes.length];
    for (int i = 0; i < selectedDTypes.length; i++) {
      typeIds[i] = selectedDTypes[i].getTypeId().getNativeId();
      scales[i] = selectedDTypes[i].getScale();
    }
    return new Table(decodeCachedLshfProjectedNative(lshfBytes, selectedCols, typeIds, scales));
  }

  public static final class StarJoinBuildTable implements AutoCloseable {
    private long handle;

    private StarJoinBuildTable(long handle) {
      this.handle = handle;
    }

    public long getHandle() {
      return handle;
    }

    @Override
    public void close() {
      if (handle != 0L) {
        long h = handle;
        handle = 0L;
        destroyStarJoinTableNative(h);
      }
    }
  }

  public static StarJoinBuildTable buildStarJoinTable(ColumnView keyCol) {
    return new StarJoinBuildTable(buildStarJoinTableNative(keyCol.getNativeView()));
  }

  public static long runStarJoinCount(
      Table factTable, int[] probeOrds, StarJoinBuildTable[] buildTables, int[] extraNotNull) {
    long[] bHandles = new long[buildTables.length];
    for (int i = 0; i < buildTables.length; i++) {
      bHandles[i] = buildTables[i].getHandle();
    }
    return runStarJoinCountNative(factTable.getNativeView(), probeOrds, bHandles, extraNotNull);
  }

  public static Table runStarJoinMaterialize(
      Table factTable,
      int rowStart,
      int rowEnd,
      int[] probeOrds,
      StarJoinBuildTable[] buildTables,
      Table[] dimTables,
      int[] extraNotNull,
      int[] outSrcTable,
      int[] outSrcCol) {
    long[] bHandles = new long[buildTables.length];
    long[] dHandles = new long[dimTables.length];
    for (int i = 0; i < buildTables.length; i++) {
      bHandles[i] = buildTables[i].getHandle();
      dHandles[i] = (dimTables[i] != null) ? dimTables[i].getNativeView() : 0L;
    }
    return new Table(runStarJoinMaterializeNative(
        factTable.getNativeView(), rowStart, rowEnd, probeOrds, bHandles, dHandles, extraNotNull,
        outSrcTable, outSrcCol));
  }

  private static native long decodeDictionaryNative(long colView);

  private static native long remapDictionaryToCanonicalNative(long colView, String[] literals);

  private static native long[] splitAndEncodeShuffleNative(long tableView, int[] splits);

  private static native long[] assembleAndDecodeShuffleNative(
      long hostAddr, long dataDevAddr, long dataDevLen, long[] hostOffsets,
      int[] typeIds, int[] scales, byte[] dict32);

  private static native long[] decodeCachedLshfProjectedNative(
      byte[] lshfBytes, int[] selectedCols, int[] typeIds, int[] scales);

  private static native long compileJitReduceNative(String cudaSrc, String kernelName);

  private static native void destroyJitReduceNative(long progHandle);

  private static native long[] runJitReduceTableNative(
      long progHandle, long tableView, String[][] dictLits,
      int[] groupKeySlots, int maxGroups, byte[] aggKinds);

  private static native long[] runJitReduceHighCardNative(
      long progHandle, long gkView, long valView, String[][] dictLits, byte[] aggKinds);

  private static native long buildStarJoinTableNative(long keyColView);

  private static native void destroyStarJoinTableNative(long handle);

  private static native long runStarJoinCountNative(
      long factTableView, int[] probeOrds, long[] buildTableHandles, int[] extraNotNull);

  private static native long[] runStarJoinMaterializeNative(
      long factTableView, int rowStart, int rowEnd, int[] probeOrds, long[] buildTableHandles,
      long[] dimTableViews, int[] extraNotNull, int[] outSrcTable, int[] outSrcCol);
}
