# Extending

A defense subclasses `BaseDefender`, which trains its model with `TrainManager`. If the defense changes how the model
answers queries, `defend_model` returns a `DefendedModel`. An attack subclasses `BaseMIA`. See
[Documenting a method](../../contributing.md) for the rest.

```{include} ../../_generated/api_extending.md
```
