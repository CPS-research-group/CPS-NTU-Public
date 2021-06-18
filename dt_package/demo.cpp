#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

def main() {
  ifstream inFile("test.jpg", ios::binary);
  std::string iBuf;
  iBuf << inFile.rdbuf();
  cout "File contents:\n" << iBuf << "\n";
  return 0;
};
