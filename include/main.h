#pragma once

#include <string>

/**
 * Display usage information for the application
 * @param programName The name of the executable
 */
void printUsage(const char* programName);

/**
 * Main entry point for the CUDA Video Cleaner application
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return Exit code (0 for success, non-zero for failure)
 */
int main(int argc, char* argv[]); 