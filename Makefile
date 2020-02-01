CC = mpicxx
FLAGS = -Ofast
all: clean
	$(CC) $(CPPFLAGS) $(FLAGS) main.cpp
clean:
	rm -rf *.o