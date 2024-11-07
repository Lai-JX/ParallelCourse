# 定义编译器
CC = gcc

# 定义编译选项
CFLAGS = -fopenmp

# 定义链接选项
LDFLAGS =

# 定义输出的可执行文件名和源文件
TARGET1 = matrix
SRC1 = matrix.c

TARGET2 = hist
SRC2 = hist.c

# 默认规则（可以选择不编译）
all: $(TARGET1) $(TARGET2)

# 编译 hist 规则
$(TARGET1): $(SRC1)
	$(CC) $(CFLAGS) -o $(TARGET1) $(SRC1) $(LDFLAGS)

# 编译 matrix 规则
$(TARGET2): $(SRC2)
	$(CC) $(CFLAGS) -o $(TARGET2) $(SRC2) $(LDFLAGS)


# 运行metrix
run1: $(TARGET1)
	./$(TARGET1)

# 运行hist
run2: $(TARGET2)
	./$(TARGET2)

# 清理规则
clean:
	rm -f $(TARGET)

