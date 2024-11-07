# 定义编译器
CC = nvcc

# 定义编译选项
CFLAGS = 

# 定义链接选项
LDFLAGS =

# 定义输出的可执行文件名和源文件
TARGET1 = p1
SRC1 = p1.cu

TARGET2 = p2
SRC2 = p2.cu

TARGET3 = p3
SRC3 = p3.cu

# 默认规则（可以选择不编译）
all: $(TARGET1) $(TARGET2)

# 编译 Matrix-Vector Multiplication 规则
$(TARGET1): $(SRC1)
	$(CC) $(CFLAGS) -o $(TARGET1) $(SRC1) $(LDFLAGS)

# 编译 Matrix Transpose 规则
$(TARGET2): $(SRC2)
	$(CC) $(CFLAGS) -o $(TARGET2) $(SRC2) $(LDFLAGS)

# 编译 convolution
$(TARGET3): $(SRC3)
	$(CC) $(CFLAGS) -o $(TARGET3) $(SRC3) $(LDFLAGS)

# 编译 convolution
tmp: tmp.cu
	$(CC) $(CFLAGS) -o tmp tmp.cu $(LDFLAGS)
run: tmp
	./tmp


# 运行 Matrix-Vector Multiplication
run1: $(TARGET1)
	./$(TARGET1)

# 运行 Matrix Transpose
run2: $(TARGET2)
	./$(TARGET2)

# 运行 convolution
run3: $(TARGET3)
	./$(TARGET3)

# 清理规则
clean:
	rm -f $(TARGET)

