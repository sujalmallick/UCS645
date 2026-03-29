cat > Makefile << 'EOF'
CC = mpicc
TARGETS = ring_comm array_sum max_min dot_product

all: $(TARGETS)

ring_comm: ring_comm.cpp
	$(CC) -o ring_comm ring_comm.cpp

array_sum: array_sum.cpp
	$(CC) -o array_sum array_sum.cpp

max_min: max_min.cpp
	$(CC) -o max_min max_min.cpp

dot_product: dot_product.cpp
	$(CC) -o dot_product dot_product.cpp

clean:
	rm -f $(TARGETS)
EOF
