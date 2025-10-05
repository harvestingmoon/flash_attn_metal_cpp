#!/bin/bash
# Build script for Flash Attention Metal on Apple Silicon
# Usage: ./build.sh [clean|rebuild|test]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Flash Attention Metal Build Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to clean build artifacts
clean_build() {
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    
    # Remove build directory
    if [ -d "build" ]; then
        rm -rf build
        echo "  * Removed build/"
    fi
    
    # Remove compiled Python extension
    find . -name "_flash_attn_metal*.so" -delete
    echo "  * Removed *.so files"
    
    # Remove Python cache
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "  * Removed Python cache"
    
    # Remove object files
    find . -name "*.o" -delete
    echo "  * Removed *.o files"
    
    echo -e "${GREEN}Clean complete!${NC}\n"
}

# Function to build
build() {
    echo -e "${BLUE}Building Flash Attention Metal...${NC}"
    
    # Check for CMake cache conflicts and clean if necessary
    if [ -f "build/CMakeCache.txt" ]; then
        echo -e "${YELLOW}Checking CMake cache...${NC}"
        # Check if cache points to different directory
        if grep -q "CMAKE_HOME_DIRECTORY" build/CMakeCache.txt; then
            cached_dir=$(grep "CMAKE_HOME_DIRECTORY" build/CMakeCache.txt | cut -d'=' -f2)
            current_dir=$(pwd)
            if [ "$cached_dir" != "$current_dir" ]; then
                echo -e "${YELLOW}CMake cache conflict detected. Cleaning build directory...${NC}"
                rm -rf build
            fi
        fi
    fi
    
    # Create build directory
    mkdir -p build
    
    # Check for Xcode Command Line Tools
    if ! command -v xcrun &> /dev/null; then
        echo -e "${YELLOW}Warning: Xcode Command Line Tools not found.${NC}"
        echo -e "${YELLOW}Some Metal shader compilation may be skipped.${NC}"
        echo -e "${YELLOW}To install: xcode-select --install${NC}\n"
    fi
    
    # Configure with CMake
    echo -e "${YELLOW}Configuring CMake...${NC}"
    if ! cmake -B build -DCMAKE_BUILD_TYPE=Release; then
        echo -e "${RED}ERROR: CMake configuration failed!${NC}"
        echo -e "${RED}Try running: ./build.sh clean${NC}"
        exit 1
    fi
    
    # Build
    echo -e "${YELLOW}Compiling...${NC}"
    if ! cmake --build build --parallel $(sysctl -n hw.ncpu); then
        echo -e "${RED}ERROR: Compilation failed!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Build complete!${NC}\n"
}

# Function to run tests
run_tests() {
    echo -e "${BLUE}Running tests...${NC}\n"
    
    if [ ! -f "test.py" ]; then
        echo -e "${RED}ERROR: test.py not found!${NC}"
        exit 1
    fi
    
    python3 test.py
    
    echo -e "\n${GREEN}Tests complete!${NC}\n"
}

# Function to verify correctness
verify() {
    echo -e "${BLUE}Verifying correctness...${NC}\n"
    
    if [ ! -f "verify_correctness.py" ]; then
        echo -e "${RED}ERROR: verify_correctness.py not found!${NC}"
        exit 1
    fi
    
    python3 verify_correctness.py
    
    echo -e "\n${GREEN}Verification complete!${NC}\n"
}

# Main script logic
case "${1:-build}" in
    clean)
        clean_build
        ;;
    rebuild)
        clean_build
        build
        ;;
    build)
        build
        ;;
    test)
        build
        run_tests
        ;;
    verify)
        build
        verify
        ;;
    all)
        clean_build
        build
        verify
        run_tests
        ;;
    help|--help|-h)
        echo "Usage: ./build.sh [command]"
        echo ""
        echo "Commands:"
        echo "  build     - Build the project (default)"
        echo "  clean     - Clean build artifacts"
        echo "  rebuild   - Clean and build"
        echo "  test      - Build and run tests"
        echo "  verify    - Build and verify correctness"
        echo "  all       - Clean, build, verify, and test"
        echo "  help      - Show this help message"
        echo ""
        exit 0
        ;;
    *)
        echo -e "${RED}ERROR: Unknown command: $1${NC}"
        echo "Run './build.sh help' for usage information"
        exit 1
        ;;
esac

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${GREEN}========================================${NC}"
