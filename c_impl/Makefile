CC ?= clang
CXX ?= clang++
CFLAGS ?= -O3 -march=native -std=c17 -Wall -Wextra -pedantic
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=
SIM_TARGET := rdim_model_simulations
GRAD_TARGET := rdim_model_gradient

SIM_SRC_C := rdim_model_simulations.c
GRAD_SRC_C := rdim_model_gradient.c
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

SIM_OBJ := $(SIM_SRC_C:.c=.o) $(SRC_CPP:.cpp=.o)
GRAD_OBJ := $(GRAD_SRC_C:.c=.o)

.PHONY: all clean

all: $(SIM_TARGET) $(GRAD_TARGET)

$(SIM_TARGET): $(SIM_OBJ)
	$(LINK) $(SIM_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

$(GRAD_TARGET): $(GRAD_OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(GRAD_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f $(SIM_TARGET) $(GRAD_TARGET) $(SIM_OBJ) $(GRAD_OBJ)
