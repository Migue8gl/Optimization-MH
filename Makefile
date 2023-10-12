CXX = clang++
CXXFLAGS = -std=c++2a -O2 -I./include -pthread -fsanitize=address,leak -g -fno-omit-frame-pointer
SRCDIR = src
FILEDIR = files
INCDIR = include
SCRPDIR = scripts
DATADIR = data
OBJDIR = obj
BINDIR = bin

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))
EXEC = $(BINDIR)/ga
TARFILE = ga.tar.gz

.PHONY: all clean directories compress

all: directories $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) -Wall -Wextra $(CXXFLAGS) $(OBJS) -o $(EXEC)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) -Wall -Wextra $(CXXFLAGS) -c $< -o $@

directories:
	mkdir -p $(OBJDIR) $(BINDIR)

clean:
	rm -rf $(OBJDIR) $(BINDIR)

compress:
	tar -czvf $(TARFILE) $(SRCDIR) $(INCDIR) $(FILEDIR) $(SCRPDIR) $(DATADIR) Makefile README.txt

