CC ?= clang
CXX ?= clang++
CFLAGS ?= -O3 -march=native -std=c17 -Wall -Wextra -pedantic
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=
TARGET := rdim_model_simulations

SRC_C := rdim_model_simulations.c
SRC_CPP :=

GUI ?= 0
IMGUI_DIR ?= third_party/imgui

ifeq ($(OPENMP),1)
CFLAGS += -fopenmp
CPPFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif

LINK := $(CC)

ifeq ($(GUI),1)
CPPFLAGS += -DWITH_GUI
SRC_CPP += viewer_imgui.cpp \
	$(IMGUI_DIR)/imgui.cpp \
	$(IMGUI_DIR)/imgui_draw.cpp \
	$(IMGUI_DIR)/imgui_tables.cpp \
	$(IMGUI_DIR)/imgui_widgets.cpp \
	$(IMGUI_DIR)/backends/imgui_impl_sdl2.cpp \
	$(IMGUI_DIR)/backends/imgui_impl_sdlrenderer2.cpp

SDL2_CFLAGS ?= $(shell sdl2-config --cflags 2>/dev/null)
SDL2_LIBS ?= $(shell sdl2-config --libs 2>/dev/null)
CXXFLAGS += $(SDL2_CFLAGS) -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends
LDFLAGS += $(SDL2_LIBS)
LINK := $(CXX)
endif

OBJ := $(SRC_C:.c=.o) $(SRC_CPP:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(LINK) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJ)
