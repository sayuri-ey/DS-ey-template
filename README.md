# Data Science Template

[![add a descriptive image, this is a good resource](.img/undraw_Science_re_mnnr.png)](https://undraw.co/)

## Business question:
TBD

## Model:
TBD

## Repository structure:

The structure includes four basic steps for a pipeline, preprocess, train, evaluate and inference. I also included some jupyter notebooks, configuration, documentation and tests directories, github workflows and makefile to make the developing and versioning process easier. Data, model, output and img are extra directories to store any input/output files.

## Setting up environment:

In order to execute all notebooks and scripts properly, the first thing would be executing the `install` command from Makefile

```
make install
```

This command will install all dependencies from `requirements.txt` and set the `virtualenv`.

## Jupytert notebooks

The template includes three main notebooks: `EDA.ipynb`, `experiment.ipynb`, and `pipeline.ipynb`. The `EDA.ipynb` is meant for all the Exploratory Data Analysis. I have also included `experiment.ipynb` notebook in which the experimenting phase can be conducted. The `pipeline.ipynb` notebook intends to simulate a pipeline flow, which executes and displays the output from the scripts in pipeline directory.
Inside the pipeline directory, another four notebooks are include, one for each of the pipeline steps: preprocess, train, evaluate, and inference. They include the functions definition for each step and may be also used for some testing and developing. The python scripts found along the jupyter notebooks are the output from the conversion made using `nbconvert`. After editing any of the four notebooks, the `script` command from the Makefile may be execute in order to trigger a new conversion.

```
make scripts
```

Organize scripts imports and format code using `isort` and `black` respectively. 
```
# Standard Libraries

# Third-Party Libraries

# Local Imports
```

# Tests
TBD

## Github workflows

I have included three workflows to help me versioning my projects. There is a `pull_request` workflow, which runs everytime a pull request is opened/synchronized/reopened and labels the size of the pull requests opened and in the future will include unit tests in order to ensure nothing breaks the pipeline. The `release` workflow also runs automatically after a merge and uses a semantic release action to identify the type of branch/commits and automatically generates the `CHANGELOG.md` and keeps track of all releases from the repository. I also included a `undo_push` to return any changes made by the last pull request, this workflows is triggered manually.
I have also included a pull request template message in order to ensure all PR best practices are followed.
The `setup.cfg` and `setup.py` scripts are used for the `release` workflow.

## Semantic release

In order for the semantic release to work, branches need to be created with a prefix, prefix/branch-name, for example, fix/pipeline-bug-fix. Commit can also include prefixes, prefix: message, for example, fix: removed bug from code.
The prefixes that may be used are `fix`, `refactor`, `feat`, `test`, `docs`, and `chore`.

fix: Use this prefix when making changes to fix a bug or resolve an issue in the project. It indicates that the changes are intended to address specific problems.
Example commit message: fix: Resolve issue with metrics plotting

refactor: This prefix is used when making code changes that improve the structure, readability, or maintainability without altering the external behavior of the code. It indicates that the changes are internal refactorings.
Example commit message: refactor: Reorganize data collection queries for improved efficiency

test: Use this prefix when making changes to test cases, test infrastructure, or adding new tests. It indicates that the changes are related to the testing process.
Example commit message: test: Add unit tests for preprocess

feat: This prefix is used when adding a new feature or significant enhancement to your codebase.
Example commit message: feat: Implement inference function

docs: Use this prefix when you're making changes to documentation, such as updating README files, adding comments, or improving code documentation.
Example commit message: docs: Update documentation for literature review

chore: This prefix is often used for routine tasks, maintenance, or other non-functional changes that don't directly affect the codebase.
Example commit message: chore: Update dependencies to the latest versions

# Documentation

Besides the `README.md`, I also included a `lit-review.md` to include any relevant information I gathered about the business question and model developed to solve it.

# Next steps:

-code formatting and style
-implement unit tests and linting in `pull_request` workflow
-add metrics monitoring
-experiment tracking
-model serving
-model versioning
-data versioning
-dockerization