# Generating Documentation Locally

The full documentation can be generated locally using the `Makefile` in the `docs` directory. We utilize Sphinx and MyST parser in addition to other dependencies. To make documentation as easy as possible to build, we provide a yaml file, which allows one to easily construct a Mamba environment with all necessary dependencies. To create the Mamba environment `docs-env`, change to the `docs` directory and execute the following:
```
mamba env create -f docs_environment.yaml
```
This new environment can then be activated as follows
```
mamba activate docs-env
```

Using the `docs-env` environment we can now easily generate the documentation locally using the following make command:
```
make html
```
this command builds the documentation and stores it in the directory `build`. 

One can then view the documentation using:
```
make view 
```
which will open the index page of the documentation. 

To remove all files generated by the make command, execute the following: 
```
make clean
```