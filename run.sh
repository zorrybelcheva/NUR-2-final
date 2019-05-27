#!/bin/bash

echo ""
echo "Running NUR second hand-in exercise solutions"
echo "Z. Belcheva, student ID: 2418797"
echo ""

echo "Check if plotting directory exists:"
if [ ! -d "plots" ]; then
  echo "Doe not exist; created."
  mkdir plots
fi

echo "Check if output directory exists:"
if [ ! -d "output" ]; then
  echo "Doe not exist; created."
  mkdir output
fi

echo "Running exercise 1a: RNG tests"
python3 -W ignore ex1a.py > output/ex1a.txt

echo "Running exercise 1b: Box-Muller transforms"
python3 -W ignore ex1b.py > output/ex1b.txt

echo "Running exercise 1c: KS-test"
python3 -W ignore ex1c.py > output/ex1c.txt

echo "Running exercise 1d: Kuiper test"
python3 -W ignore ex1d.py > output/ex1d.txt

echo "Running exercise 1e: Testing 10 datasets"
if [ ! -e randomnumbers.txt ]; then
  wget https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt 
fi
echo "Data set downloaded"
python3 -W ignore ex1e.py > output/ex1e.txt

echo "Running exercise 2: Initial density field"
python3 -W ignore ex2.py > output/ex2.txt

echo "Running exercise 3: Linear structure growth"
python3 -W ignore ex3.py > output/ex3.txt

echo "Running exercise 4a: Zeldovich approximation, calculating linear growth factor"
python3 -W ignore ex4a.py > output/ex4a.txt

echo "Running exercise 4b: Linear growth factor, first time derivative"
python3 -W ignore ex4b.py > output/ex4b.txt

echo "Running exercise 5a: Mass assignment schemes"
python3 -W ignore ex5a.py > output/ex5a.txt

echo "Running exercise 5 supplementary"
python3 -W ignore ex5a_supplementary.py > output/ex5a-supplementary.txt

echo "Running exercise 5b: Individual cells"
python3 -W ignore ex5b.py > output/ex5b.txt

echo "Running exercise 7: Building a Barnes-Hut quadtree"
if [ ! -e colliding.hdf5 ]; then
  wget https://home.strw.leidenuniv.nl/~nobels/coursedata/colliding.hdf5
fi
echo "Data set downloaded"
python3 -W ignore ex7.py > output/ex7.txt

echo "Generating .pdf of report"
pdflatex report.tex
pdflatex report.tex

echo "Done!"
