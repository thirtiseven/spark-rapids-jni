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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/**
 * Thin JNI bridge to libcudf's Whole-Stage GPU Codegen primitives (cudf::whole_stage):
 * pluggable Sources (Raw / Dict / LSHF / Vortex / FSST ByteStream), L2 LUT/Hash Broadcast Probes,
 * and Native Single-Pass GPU Open-Addressing Hash-Agg Sink (with auto-flush OOM safety).
 */
public final class WholeStageNative {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private WholeStageNative() {}

  public static final int PREAMBLE_FOR256    = 1 << 0;
  public static final int PREAMBLE_VORTEX_BP = 1 << 1;
  public static final int PREAMBLE_FSST      = 1 << 2;
  public static final int PREAMBLE_SMEM      = 1 << 3;
  public static final int PREAMBLE_MURMUR3   = 1 << 4;
  public static final int PREAMBLE_ALL       = 0x1F;

  public static final class Program implements AutoCloseable {
    private long handle;

    private Program(long handle) {
      this.handle = handle;
    }

    public HashAggTask openHashAggTask(DType[] keyTypes, byte[] aggKinds, int initialCapacity) {
      return openHashAggTask(keyTypes, aggKinds, initialCapacity, null);
    }

    public HashAggTask openHashAggTask(
        DType[] keyTypes, byte[] aggKinds, int initialCapacity, long[] constBits) {
      int[] tids = new int[keyTypes.length];
      int[] scales = new int[keyTypes.length];
      for (int i = 0; i < keyTypes.length; i++) {
        tids[i] = keyTypes[i].getTypeId().getNativeId();
        scales[i] = keyTypes[i].getScale();
      }
      long h = openHashAggTaskNative(handle, tids, scales, aggKinds, initialCapacity, constBits);
      return new HashAggTask(h);
    }

    @Override
    public void close() {
      if (handle != 0L) {
        long h = handle;
        handle = 0L;
        destroyProgramNative(h);
      }
    }
  }

  public static final class ProbeTable implements AutoCloseable {
    private long handle;

    private ProbeTable(long handle) {
      this.handle = handle;
    }

    public long getNativeHandle() {
      return handle;
    }

    @Override
    public void close() {
      if (handle != 0L) {
        long h = handle;
        handle = 0L;
        destroyProbeTableNative(h);
      }
    }
  }

  public static final class HashAggTask implements AutoCloseable {
    private long handle;

    private HashAggTask(long handle) {
      this.handle = handle;
    }

    public long getNativeHandle() {
      return handle;
    }

    public void addBatch(
        ColumnView[] factCols,
        int numRows,
        long[] probeHandles,
        ColumnView[] dimCols) {
      long[] fViews = new long[factCols.length];
      for (int i = 0; i < factCols.length; i++) {
        fViews[i] = factCols[i].getNativeView();
      }
      long[] dViews = new long[dimCols.length];
      for (int i = 0; i < dimCols.length; i++) {
        dViews[i] = dimCols[i].getNativeView();
      }
      addBatchHashAggNative(handle, fViews, numRows, probeHandles, dViews);
    }

    public Table finish() {
      return new Table(finishHashAggTaskNative(handle));
    }

    @Override
    public void close() {
      if (handle != 0L) {
        long h = handle;
        handle = 0L;
        destroyHashAggTaskNative(h);
      }
    }
  }

  public static Program compileProgram(String bodyCudaSrc, String kernelName) {
    return compileProgram(bodyCudaSrc, kernelName, PREAMBLE_ALL);
  }

  public static Program compileProgram(String bodyCudaSrc, String kernelName, int preambleFlags) {
    return new Program(compileProgramNative(bodyCudaSrc, kernelName, preambleFlags));
  }

  /**
   * Pure-CPU NVRTC compilation to arch-independent PTX (default compute_75). Safe on CPU-only
   * Spark Drivers (no GPU / CUDA driver required). Returns null if native compile is unavailable.
   */
  public static byte[] tryCompileToPtxCpuOnly(
      String bodyCudaSrc, String kernelName, int preambleFlags) {
    try {
      return compileToPtxCpuOnlyNative(bodyCudaSrc, kernelName, preambleFlags, "compute_75");
    } catch (Throwable t) {
      return null;
    }
  }

  public static Program loadProgramFromPtx(byte[] ptxBytes, String kernelName) {
    return new Program(loadProgramFromPtxNative(ptxBytes, kernelName));
  }

  public static ProbeTable buildProbeTable(ColumnView keyCol) {
    return new ProbeTable(buildProbeTableNative(keyCol.getNativeView()));
  }

  /** Returns [codes: INT32 (preserving input nulls), keys: STRING]. substrLen < 0 means full string. */
  public static ai.rapids.cudf.ColumnVector[] encodeDimStringDict(
      ColumnView strCol, int substrStart0, int substrLen) {
    long[] ptrs = encodeDimStringDictNative(strCol.getNativeView(), substrStart0, substrLen);
    return new ai.rapids.cudf.ColumnVector[] {
      new ai.rapids.cudf.ColumnVector(ptrs[0]),
      new ai.rapids.cudf.ColumnVector(ptrs[1])
    };
  }

  public static boolean tryAddLshfFrameToAgg(
      HashAggTask aggTask,
      long hostAddr,
      long hostLen,
      long devAddr,
      long devLen,
      int[] selectedCols,
      long[] probeHandles,
      ColumnView[] dimCols) {
    long[] dViews = new long[dimCols.length];
    for (int i = 0; i < dimCols.length; i++) {
      dViews[i] = dimCols[i].getNativeView();
    }
    return tryAddLshfFrameToAggNative(
        aggTask.getNativeHandle(), hostAddr, hostLen, devAddr, devLen, selectedCols, probeHandles, dViews);
  }

  public static Table runStagePartitionScatter(
      Program countProg,
      Program scatterProg,
      ColumnView[] factCols,
      int numRows,
      long[] probeHandles,
      ColumnView[] dimCols,
      DType[] outTypes,
      int numParts,
      int[] outSplits) {
    return runStagePartitionScatter(
        countProg, scatterProg, factCols, numRows, probeHandles, dimCols, outTypes, numParts, outSplits, null);
  }

  public static Table runStagePartitionScatter(
      Program countProg,
      Program scatterProg,
      ColumnView[] factCols,
      int numRows,
      long[] probeHandles,
      ColumnView[] dimCols,
      DType[] outTypes,
      int numParts,
      int[] outSplits,
      long[] constBits) {
    long[] fViews = new long[factCols.length];
    for (int i = 0; i < factCols.length; i++) { fViews[i] = factCols[i].getNativeView(); }
    long[] dViews = new long[dimCols.length];
    for (int i = 0; i < dimCols.length; i++) { dViews[i] = dimCols[i].getNativeView(); }
    int[] tids = new int[outTypes.length];
    int[] scales = new int[outTypes.length];
    for (int i = 0; i < outTypes.length; i++) {
      tids[i] = outTypes[i].getTypeId().getNativeId();
      scales[i] = outTypes[i].getScale();
    }
    return new Table(runStagePartitionScatterNative(
        countProg.handle, scatterProg.handle, fViews, numRows, probeHandles, dViews,
        tids, scales, numParts, outSplits, constBits));
  }

  public static Table runStageMaterialize(
      Program countProg,
      Program scatterProg,
      ColumnView[] factCols,
      int numRows,
      long[] probeHandles,
      ColumnView[] dimCols,
      DType[] outTypes) {
    return runStageMaterialize(
        countProg, scatterProg, factCols, numRows, probeHandles, dimCols, outTypes, null);
  }

  public static Table runStageMaterialize(
      Program countProg,
      Program scatterProg,
      ColumnView[] factCols,
      int numRows,
      long[] probeHandles,
      ColumnView[] dimCols,
      DType[] outTypes,
      long[] constBits) {
    long[] fViews = new long[factCols.length];
    for (int i = 0; i < factCols.length; i++) { fViews[i] = factCols[i].getNativeView(); }
    long[] dViews = new long[dimCols.length];
    for (int i = 0; i < dimCols.length; i++) { dViews[i] = dimCols[i].getNativeView(); }
    int[] tids = new int[outTypes.length];
    int[] scales = new int[outTypes.length];
    for (int i = 0; i < outTypes.length; i++) {
      tids[i] = outTypes[i].getTypeId().getNativeId();
      scales[i] = outTypes[i].getScale();
    }
    return new Table(runStageMaterializeNative(
        countProg.handle, scatterProg.handle, fViews, numRows, probeHandles, dViews, tids, scales, constBits));
  }

  private static native long compileProgramNative(
      String bodyCudaSrc, String kernelName, int preambleFlags);
  private static native byte[] compileToPtxCpuOnlyNative(
      String bodyCudaSrc, String kernelName, int preambleFlags, String computeArch);
  private static native long loadProgramFromPtxNative(byte[] ptxBytes, String kernelName);
  private static native void destroyProgramNative(long progHandle);
  private static native long buildProbeTableNative(long keyColView);
  private static native void destroyProbeTableNative(long probeHandle);
  private static native long openHashAggTaskNative(
      long progHandle, int[] keyTypeIds, int[] keyScales, byte[] aggKinds, int initCap, long[] constBits);
  private static native void addBatchHashAggNative(
      long stateHandle, long[] factColViews, int numRows, long[] probeHandles, long[] dimColViews);
  private static native long[] finishHashAggTaskNative(long stateHandle);
  private static native void destroyHashAggTaskNative(long stateHandle);
  private static native long[] encodeDimStringDictNative(
      long colView, int substrStart0, int substrLen);
  private static native boolean tryAddLshfFrameToAggNative(
      long stateHandle, long hostAddr, long hostLen, long devAddr, long devLen,
      int[] selCols, long[] probeHandles, long[] dimColViews);
  private static native long[] runStagePartitionScatterNative(
      long countProg, long scatterProg, long[] factCols, int numRows,
      long[] probeHandles, long[] dimCols, int[] outTids, int[] outScales,
      int numParts, int[] outSplits, long[] constBits);
  private static native long[] runStageMaterializeNative(
      long countProg, long scatterProg, long[] factCols, int numRows,
      long[] probeHandles, long[] dimCols, int[] outTids, int[] outScales, long[] constBits);
}
