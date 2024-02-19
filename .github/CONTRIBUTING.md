# Contributing

We appreciate your interest in contributing. To ensure a smooth collaboration, please review the following guidelines.

## How to Contribute

1. Get the latest version of the repository:
    - For the first time: Fork the repository. Clone the forked repository to your local machine.
    - For the second time: Sync your fork with the main repository.
2. Create a new branch for your changes:
    ```bash
    git checkout -b feature/new-feature
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Add a concise commit message"
    ```
4. Push your changes to your fork:
    ```bash
    git push origin feature/new-feature
    ```
5. Open a pull request to the main repository on the `main` branch.

## Code Style

- Use [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort) to reorder import statements
- Use [yapf](https://marketplace.visualstudio.com/items?itemName=eeyore.yapf) to format Python codes into Google style
 - We use `based_on_style = "google"`. See https://google.github.io/styleguide/pyguide.html
- Use [Google Docstring Format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) to standardize docstrings
- Use [Conventional Commits](https://www.conventionalcommits.org/) to make commit messages more readable
