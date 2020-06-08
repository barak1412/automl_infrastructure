cd source
sphinx-apidoc --force -d 1 -o . ../../../automl_infrastructure
cd ..
make html