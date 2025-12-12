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
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * Unit tests for ProtobufUtils and ProtobufOptions.
 */
public class ProtobufUtilsTest {

    /**
     * Test creating ProtobufUtils options builder.
     */
    @Test
    public void testOptionsBuilder() {
        ProtobufUtils.OptionsBuilder builder = ProtobufUtils.builder()
            .withField("logid", 1, DType.INT64)
            .withField("date", 2, DType.INT64)
            .withField("user_ip", 5, DType.STRING)
            .withHadoopSequenceFile(true);
        
        assertEquals(3, builder.getSchema().size());
        assertTrue(builder.isHadoopSequenceFile());
        
        // Verify field info
        assertEquals("logid", builder.getSchema().get(0).getName());
        assertEquals(1, builder.getSchema().get(0).getFieldNumber());
        assertEquals(DType.INT64, builder.getSchema().get(0).getDType());
    }

    /**
     * Test ProtobufOptions builder.
     */
    @Test
    public void testProtobufOptionsBuilder() {
        ProtobufOptions opts = ProtobufOptions.builder()
            .withField("id", 1, DType.INT64)
            .withField("name", 2, DType.STRING)
            .withField("value", 3, DType.FLOAT64)
            .withHadoopSequenceFile(true)
            .build();
        
        assertEquals(3, opts.getNumFields());
        assertTrue(opts.isHadoopSequenceFile());
        
        // Check arrays
        String[] names = opts.getFieldNames();
        assertEquals("id", names[0]);
        assertEquals("name", names[1]);
        assertEquals("value", names[2]);
        
        int[] fieldNumbers = opts.getFieldNumbers();
        assertEquals(1, fieldNumbers[0]);
        assertEquals(2, fieldNumbers[1]);
        assertEquals(3, fieldNumbers[2]);
    }

    /**
     * Test FieldInfo creation helpers.
     */
    @Test
    public void testFieldInfoHelpers() {
        ProtobufUtils.FieldInfo int64Field = ProtobufUtils.int64Field("count", 1);
        assertEquals("count", int64Field.getName());
        assertEquals(1, int64Field.getFieldNumber());
        assertEquals(DType.INT64, int64Field.getDType());

        ProtobufUtils.FieldInfo stringField = ProtobufUtils.stringField("name", 2);
        assertEquals("name", stringField.getName());
        assertEquals(2, stringField.getFieldNumber());
        assertEquals(DType.STRING, stringField.getDType());

        ProtobufUtils.FieldInfo float64Field = ProtobufUtils.float64Field("value", 3);
        assertEquals("value", float64Field.getName());
        assertEquals(3, float64Field.getFieldNumber());
        assertEquals(DType.FLOAT64, float64Field.getDType());

        ProtobufUtils.FieldInfo float32Field = ProtobufUtils.float32Field("score", 4);
        assertEquals("score", float32Field.getName());
        assertEquals(4, float32Field.getFieldNumber());
        assertEquals(DType.FLOAT32, float32Field.getDType());

        ProtobufUtils.FieldInfo boolField = ProtobufUtils.boolField("active", 5);
        assertEquals("active", boolField.getName());
        assertEquals(5, boolField.getFieldNumber());
        assertEquals(DType.BOOL8, boolField.getDType());

        ProtobufUtils.FieldInfo int32Field = ProtobufUtils.int32Field("age", 6);
        assertEquals("age", int32Field.getName());
        assertEquals(6, int32Field.getFieldNumber());
        assertEquals(DType.INT32, int32Field.getDType());
    }

    /**
     * Test that empty schema throws exception for ProtobufUtils.OptionsBuilder.
     */
    @Test
    public void testEmptySchemaThrowsForOptionsBuilder() {
        ProtobufUtils.OptionsBuilder builder = ProtobufUtils.builder();
        assertThrows(IllegalStateException.class, () -> builder.build());
    }

    /**
     * Test that empty schema throws exception for ProtobufOptions.Builder.
     */
    @Test
    public void testEmptySchemaThrowsForProtobufOptions() {
        ProtobufOptions.Builder builder = ProtobufOptions.builder();
        assertThrows(IllegalStateException.class, () -> builder.build());
    }

    /**
     * Test ProtobufOptions field info access.
     */
    @Test
    public void testProtobufOptionsFieldInfo() {
        ProtobufOptions opts = ProtobufOptions.builder()
            .withField("field1", 1, DType.INT64)
            .withField("field2", 2, DType.STRING)
            .build();
        
        assertEquals(2, opts.getFields().size());
        
        ProtobufOptions.FieldInfo field1 = opts.getFields().get(0);
        assertEquals("field1", field1.getName());
        assertEquals(1, field1.getFieldNumber());
        assertEquals(DType.INT64, field1.getDType());
        assertEquals(0, field1.getScale());
        
        ProtobufOptions.FieldInfo field2 = opts.getFields().get(1);
        assertEquals("field2", field2.getName());
        assertEquals(2, field2.getFieldNumber());
        assertEquals(DType.STRING, field2.getDType());
    }

    /**
     * Test that readProtobuf throws UnsupportedOperationException.
     * Native protobuf parsing is not yet implemented.
     */
    @Test
    public void testReadProtobufNotSupported() {
        ProtobufOptions opts = ProtobufUtils.builder()
            .withField("value", 1, DType.INT64)
            .withHadoopSequenceFile(false)
            .build();
        
        byte[] data = new byte[] {1, 2, 3, 4};
        
        // Should throw UnsupportedOperationException
        assertThrows(UnsupportedOperationException.class, 
            () -> ProtobufUtils.readProtobuf(opts, data));
    }

    /**
     * Test that readProtobuf from file throws UnsupportedOperationException.
     */
    @Test
    public void testReadProtobufFromFileNotSupported() {
        ProtobufOptions opts = ProtobufUtils.builder()
            .withField("value", 1, DType.INT64)
            .build();
        
        // Should throw UnsupportedOperationException
        assertThrows(UnsupportedOperationException.class, 
            () -> ProtobufUtils.readProtobuf(opts, "/tmp/test.pb"));
    }

    /**
     * Test that readHadoopSequenceFile throws UnsupportedOperationException.
     */
    @Test
    public void testReadHadoopSequenceFileNotSupported() {
        // Should throw UnsupportedOperationException
        assertThrows(UnsupportedOperationException.class, 
            () -> ProtobufUtils.readHadoopSequenceFile("/tmp/test.seq",
                ProtobufUtils.int64Field("id", 1),
                ProtobufUtils.stringField("name", 2)));
    }

    /**
     * Test ProtobufOptions schema list.
     */
    @Test
    public void testProtobufOptionsSchema() {
        ProtobufOptions opts = ProtobufOptions.builder()
            .withField("a", 1, DType.INT64)
            .withField("b", 2, DType.STRING)
            .withField("c", 3, DType.FLOAT64)
            .build();
        
        java.util.List<String> schema = opts.getSchema();
        assertEquals(3, schema.size());
        assertEquals("a", schema.get(0));
        assertEquals("b", schema.get(1));
        assertEquals("c", schema.get(2));
    }

    /**
     * Test ProtobufOptions with scale for decimal types.
     */
    @Test
    public void testProtobufOptionsWithScale() {
        ProtobufOptions opts = ProtobufOptions.builder()
            .withField("amount", 1, DType.create(DType.DTypeEnum.DECIMAL64, -2), 2)
            .build();
        
        assertEquals(1, opts.getNumFields());
        int[] scales = opts.getScales();
        assertEquals(2, scales[0]);
    }

    /**
     * Test Compression enum values.
     */
    @Test
    public void testCompressionEnum() {
        assertEquals(6, ProtobufUtils.Compression.values().length);
        assertNotNull(ProtobufUtils.Compression.NONE);
        assertNotNull(ProtobufUtils.Compression.GZIP);
        assertNotNull(ProtobufUtils.Compression.SNAPPY);
        assertNotNull(ProtobufUtils.Compression.LZ4);
        assertNotNull(ProtobufUtils.Compression.ZSTD);
        assertNotNull(ProtobufUtils.Compression.AUTO);
    }

    /**
     * Test OptionsBuilder with compression setting.
     */
    @Test
    public void testOptionsBuilderWithCompression() {
        ProtobufUtils.OptionsBuilder builder = ProtobufUtils.builder()
            .withField("data", 1, DType.STRING)
            .withCompression(ProtobufUtils.Compression.SNAPPY);
        
        // Build should work
        ProtobufOptions opts = builder.build();
        assertNotNull(opts);
    }
}
