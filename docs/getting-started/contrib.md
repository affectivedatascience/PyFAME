# Contributing to PyFAME

Thank you for your interest in contributing to PyFAME! Contributions
including bug fixes, new features, documentation improvements, and other
enhancements are welcome.

Development is managed through GitHub. Contributions should be made on a
separate branch and submitted through a pull request.

## Clone the Repository

First, clone the PyFAME repository and navigate into the project
directory:

``` bash
git clone https://github.com/AffectiveDataScience/PyFAME.git
cd PyFAME
```

It is recommended that development be performed within a Python virtual
environment.

=== "Windows"

    ```powershell
    py -m venv .venv
    .venv\Scripts\activate
    ```

=== "macOS / Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

## Install Dependencies

PyFAME's runtime dependencies are listed in `requirements.txt`. Install
them using:

``` bash
pip install -r requirements.txt
```

Additional packages used for development and testing are listed in
`requirements_dev.txt`:

``` bash
pip install -r requirements_dev.txt
```

After installation, your environment should contain everything required
to develop and test PyFAME locally.

## Create a Development Branch

Before making changes, create a new branch from the latest version of
the repository:

``` bash
git checkout master
git pull origin master
git checkout -b your-branch-name
```

Use a short, descriptive branch name related to your changes, for
example:

``` bash
git checkout -b fix-overlay-positioning
```

or:

``` bash
git checkout -b add-new-analysis-function
```

Make and test your changes on this branch rather than directly on
`master`.

## Commit Your Changes

Once your changes are ready, stage and commit them:

``` bash
git add .
git commit -m "Brief description of your changes"
```

Push your branch to GitHub:

``` bash
git push -u origin your-branch-name
```

After the initial push, subsequent changes can normally be pushed with:

``` bash
git push
```

## Open a Pull Request

Navigate to the [PyFAME GitHub
repository](https://github.com/AffectiveDataScience/PyFAME) after
pushing your branch. GitHub will typically display an option to
**Compare & pull request** for the recently pushed branch.

When creating the pull request:

1.  Set the **base** branch to `master`.
2.  Set the **compare/head** branch to your development branch.
3.  Give the pull request a concise, descriptive title.
4.  Describe what was changed and why.
5.  Include any information needed to test or reproduce the changes.

Submit the pull request when it is ready for review.

!!! tip "Base and head branches"

    The **base** branch is the branch that you want your changes merged **into** (`master` for PyFAME).

    The **head** or **compare** branch contains the changes that you want merged.

## Updating a Pull Request

A pull request does not need to be recreated if additional changes are
required during review.

Simply make additional commits on the same development branch:

``` bash
git add .
git commit -m "Address pull request feedback"
git push
```

The existing pull request will automatically update with the new
commits.

## Reporting Bugs

If you encounter a problem but do not intend to submit a code change,
please open an issue through the [PyFAME GitHub issue
tracker](https://github.com/AffectiveDataScience/PyFAME/issues).

When reporting a bug, include enough information to reproduce the
problem, including the relevant PyFAME and Python versions, input
parameters, and any error messages or tracebacks.
