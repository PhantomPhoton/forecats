# Contributing to Forecats

## How to Contribute

1. **Fork the repository** and create your branch from `master`.
2. **Make your changes** with a descriptive commit messages.
3. **Test your changes** to make sure nothing is broken.
4. **Submit a pull request** with a description of your changes. If you are fixing an issue please add `closes #\<issue number\>` to the PR description.

## Testing

There are no unit tests atm. You are welcome to contribute some if you want (preferrably using pytest). Otherwise, you can locally test that everything still works fine with

```bash
    cd local_testing && uv run test.py
```

Note that this requires [uv](https://docs.astral.sh/uv/).

Please try running it on an HA server before submitting the PR.

## Guidelines

- If you change the `forecats.generate_pet_picture` action fields, make sure these changes are reflected in `models.py`, `services.yaml`, and `local_testing/test.py` as well.
- Provider credentials and model/timeout settings belong in the config flow (`config_flow.py` / `const.py`), not in the service schema.

## Need Help?

If you have questions, open an issue or start a discussion.
