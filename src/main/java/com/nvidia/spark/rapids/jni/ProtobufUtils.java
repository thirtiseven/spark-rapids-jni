/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Table;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for reading Protobuf data on GPU.
 * 
 * This class provides a Spark-friendly wrapper for protobuf reading,
 * supporting Hadoop SequenceFile format commonly used in big data environments.
 * 
 * Note: Native protobuf parsing support is not yet available in cudf.
 * This class provides the API structure for future implementation.
 */
public class ProtobufUtils {
    // Note: Native deps loading will be needed when native protobuf support is implemented
    // static {
    //     NativeDepsLoader.loadNativeDeps();
    // }

    /**
     * Schema definition for a protobuf field.
     */
    public static class FieldInfo {
        private final String name;
        private final int fieldNumber;
        private final DType dtype;
        private final boolean isRepeated;

        /**
         * Create a field definition for protobuf schema.
         *
         * @param name The field name (will become column name)
         * @param fieldNumber The protobuf field number/tag
         * @param dtype The cudf data type for this field
         */
        public FieldInfo(String name, int fieldNumber, DType dtype) {
            this(name, fieldNumber, dtype, false);
        }

        /**
         * Create a field definition for protobuf schema.
         *
         * @param name The field name (will become column name)
         * @param fieldNumber The protobuf field number/tag
         * @param dtype The cudf data type for this field
         * @param isRepeated Whether this is a repeated field
         */
        public FieldInfo(String name, int fieldNumber, DType dtype, boolean isRepeated) {
            this.name = name;
            this.fieldNumber = fieldNumber;
            this.dtype = dtype;
            this.isRepeated = isRepeated;
        }

        public String getName() { return name; }
        public int getFieldNumber() { return fieldNumber; }
        public DType getDType() { return dtype; }
        public boolean isRepeated() { return isRepeated; }
    }

    /**
     * Compression type for protobuf data.
     */
    public enum Compression {
        NONE,
        GZIP,
        SNAPPY,
        LZ4,
        ZSTD,
        AUTO
    }

    /**
     * Builder for protobuf reader options.
     */
    public static class OptionsBuilder {
        private final List<FieldInfo> schema = new ArrayList<>();
        private boolean isHadoopSequenceFile = false;
        private Compression compression = Compression.AUTO;

        /**
         * Add a field to the schema.
         *
         * @param name Field name
         * @param fieldNumber Protobuf field number
         * @param dtype Data type
         * @return this builder
         */
        public OptionsBuilder withField(String name, int fieldNumber, DType dtype) {
            schema.add(new FieldInfo(name, fieldNumber, dtype));
            return this;
        }

        /**
         * Add a field to the schema.
         *
         * @param fieldInfo Field information
         * @return this builder
         */
        public OptionsBuilder withField(FieldInfo fieldInfo) {
            schema.add(fieldInfo);
            return this;
        }

        /**
         * Set whether the file is in Hadoop SequenceFile format.
         *
         * @param isHadoop true if the file is a Hadoop SequenceFile
         * @return this builder
         */
        public OptionsBuilder withHadoopSequenceFile(boolean isHadoop) {
            this.isHadoopSequenceFile = isHadoop;
            return this;
        }

        /**
         * Set the compression type.
         *
         * @param compression Compression type
         * @return this builder
         */
        public OptionsBuilder withCompression(Compression compression) {
            this.compression = compression;
            return this;
        }

        /**
         * Build the ProtobufOptions.
         *
         * @return ProtobufOptions instance
         */
        public ProtobufOptions build() {
            if (schema.isEmpty()) {
                throw new IllegalStateException("Schema must have at least one field");
            }
            
            ProtobufOptions.Builder builder = ProtobufOptions.builder();
            for (FieldInfo field : schema) {
                builder.withField(field.getName(), field.getFieldNumber(), field.getDType());
            }
            builder.withHadoopSequenceFile(isHadoopSequenceFile);
            return builder.build();
        }

        public List<FieldInfo> getSchema() { return schema; }
        public boolean isHadoopSequenceFile() { return isHadoopSequenceFile; }
    }

    /**
     * Create an options builder.
     *
     * @return new OptionsBuilder
     */
    public static OptionsBuilder builder() {
        return new OptionsBuilder();
    }

    // ===================================================================================
    // Native method declarations for future protobuf reading support.
    // These will be implemented when native protobuf parsing is added to spark-rapids-jni.
    // ===================================================================================

    /**
     * Read protobuf data from a buffer.
     * 
     * @param fieldNames Array of field names
     * @param fieldNumbers Array of protobuf field numbers
     * @param dTypeIds Array of data type IDs
     * @param dTypeScales Array of scales for decimal types
     * @param address Memory address of buffer
     * @param length Length of buffer
     * @param isHadoopSeqFile true if data is in Hadoop SequenceFile format
     * @return Array of column handles
     */
    private static native long[] readProtobuf(String[] fieldNames, int[] fieldNumbers,
                                              int[] dTypeIds, int[] dTypeScales,
                                              long address, long length,
                                              boolean isHadoopSeqFile);

    /**
     * Read protobuf data from a file.
     * 
     * @param options Protobuf reader options
     * @param file The file to read
     * @return Table containing the parsed data
     * @throws UnsupportedOperationException Native protobuf parsing is not yet implemented
     */
    public static Table readProtobuf(ProtobufOptions options, File file) {
        throw new UnsupportedOperationException(
            "Native protobuf parsing is not yet implemented. " +
            "This feature requires native GPU support for protobuf format.");
    }

    /**
     * Read protobuf data from a file path.
     *
     * @param options Protobuf reader options
     * @param filePath Path to the file
     * @return Table containing the parsed data
     * @throws UnsupportedOperationException Native protobuf parsing is not yet implemented
     */
    public static Table readProtobuf(ProtobufOptions options, String filePath) {
        return readProtobuf(options, new File(filePath));
    }

    /**
     * Read protobuf data from a byte array.
     *
     * @param options Protobuf reader options
     * @param data Raw protobuf data
     * @return Table containing the parsed data
     * @throws UnsupportedOperationException Native protobuf parsing is not yet implemented
     */
    public static Table readProtobuf(ProtobufOptions options, byte[] data) {
        throw new UnsupportedOperationException(
            "Native protobuf parsing is not yet implemented. " +
            "This feature requires native GPU support for protobuf format.");
    }

    /**
     * Read protobuf data from a host memory buffer.
     *
     * @param options Protobuf reader options
     * @param buffer Host memory buffer containing protobuf data
     * @param offset Starting offset in the buffer
     * @param length Length of data to read
     * @return Table containing the parsed data
     * @throws UnsupportedOperationException Native protobuf parsing is not yet implemented
     */
    public static Table readProtobuf(ProtobufOptions options, HostMemoryBuffer buffer, 
                                     long offset, long length) {
        throw new UnsupportedOperationException(
            "Native protobuf parsing is not yet implemented. " +
            "This feature requires native GPU support for protobuf format.");
    }

    /**
     * Convenience method to read Hadoop SequenceFile containing protobuf messages.
     *
     * @param filePath Path to the Hadoop SequenceFile
     * @param fields List of field definitions
     * @return Table containing the parsed data
     * @throws UnsupportedOperationException Native protobuf parsing is not yet implemented
     */
    public static Table readHadoopSequenceFile(String filePath, List<FieldInfo> fields) {
        OptionsBuilder builder = new OptionsBuilder();
        for (FieldInfo field : fields) {
            builder.withField(field);
        }
        builder.withHadoopSequenceFile(true);
        return readProtobuf(builder.build(), filePath);
    }

    /**
     * Convenience method to read Hadoop SequenceFile with varargs fields.
     *
     * @param filePath Path to the Hadoop SequenceFile
     * @param fields Field definitions
     * @return Table containing the parsed data
     * @throws UnsupportedOperationException Native protobuf parsing is not yet implemented
     */
    public static Table readHadoopSequenceFile(String filePath, FieldInfo... fields) {
        OptionsBuilder builder = new OptionsBuilder();
        for (FieldInfo field : fields) {
            builder.withField(field);
        }
        builder.withHadoopSequenceFile(true);
        return readProtobuf(builder.build(), filePath);
    }

    // ===================================================================================
    // Helper methods to create field definitions
    // ===================================================================================

    /**
     * Create a field definition with INT64 type.
     *
     * @param name Field name
     * @param fieldNumber Protobuf field number
     * @return FieldInfo for an INT64 field
     */
    public static FieldInfo int64Field(String name, int fieldNumber) {
        return new FieldInfo(name, fieldNumber, DType.INT64);
    }

    /**
     * Create a field definition with INT32 type.
     *
     * @param name Field name
     * @param fieldNumber Protobuf field number
     * @return FieldInfo for an INT32 field
     */
    public static FieldInfo int32Field(String name, int fieldNumber) {
        return new FieldInfo(name, fieldNumber, DType.INT32);
    }

    /**
     * Create a field definition with STRING type.
     *
     * @param name Field name
     * @param fieldNumber Protobuf field number
     * @return FieldInfo for a STRING field
     */
    public static FieldInfo stringField(String name, int fieldNumber) {
        return new FieldInfo(name, fieldNumber, DType.STRING);
    }

    /**
     * Create a field definition with FLOAT64 type.
     *
     * @param name Field name
     * @param fieldNumber Protobuf field number
     * @return FieldInfo for a FLOAT64 field
     */
    public static FieldInfo float64Field(String name, int fieldNumber) {
        return new FieldInfo(name, fieldNumber, DType.FLOAT64);
    }

    /**
     * Create a field definition with FLOAT32 type.
     *
     * @param name Field name
     * @param fieldNumber Protobuf field number
     * @return FieldInfo for a FLOAT32 field
     */
    public static FieldInfo float32Field(String name, int fieldNumber) {
        return new FieldInfo(name, fieldNumber, DType.FLOAT32);
    }

    /**
     * Create a field definition with BOOL8 type.
     *
     * @param name Field name
     * @param fieldNumber Protobuf field number
     * @return FieldInfo for a BOOL8 field
     */
    public static FieldInfo boolField(String name, int fieldNumber) {
        return new FieldInfo(name, fieldNumber, DType.BOOL8);
    }
}
