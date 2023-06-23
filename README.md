# FilterPy
### A collection of Bayesian filtering methods in Python using Numpy and Scipy

<!-- tempate https://github.com/scottydocs/README-template.md/blob/master/README.md -->
![GitHub repo size](https://img.shields.io/github/repo-size/mjcarter95/FilterPy)
![GitHub contributors](https://img.shields.io/github/contributors/mjcarter95/FilterPy)
![GitHub stars](https://img.shields.io/github/stars/mjcarter95/FilterPy?style=social)
![GitHub forks](https://img.shields.io/github/forks/mjcarter95/FilterPy?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/mjcarter955?style=social)

FilterPy allows users to filter and track state space models using various Bayesian inference methods.

## Installing FilterPy

To install FilterPy, follow these steps:

```
pip install pip@git+https://github.com/mjcarter95/FilterPy.git
```

## Using FilterPy

A number of example problems are provided in the `examples` folder.

### Example: Linear Gaussian State Space Model

```
  # Instantiate the measurement and observation models
  transition_model = model.TransitionModel(F, Q)
  measurement_model = model.MeasurementModel(H, R)

  # Simulate the state and observation sequences
  x_true, y = lgssm.simulate_data(T, transition_model, measurement_model)

  # Instantiate the Kalman filter
  kf = BasicKalmanFilter(transition_model, measurement_model)

  # Initialise the state and state covariance
  x_hat = np.zeros((T, 1))
  P = np.zeros((T, 1))

  # Set the initial state and state covariance
  x_hat[0] = np.random.multivariate_normal(np.zeros(1), np.eye(1))
  P[0] = np.array([[0.5]]])

  # Run the Kalman filter
  for t in range(1, T):
      x_pred, P_pred = kf.predict(x_hat[t-1], P[t-1])
      x_hat[t], P[t] = kf.update(x_pred, P_pred, y[t])
```

## Contributing to FilterPy
<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->
To contribute to FilterPy, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

<!-- ## Contributors

Thanks to the following people who have contributed to this project: -->

<!-- * [@mjcarter95](https://github.com/mjcarter95) 📖 -->
<!-- * [@vberaud](https://github.com/vberaud) 🐛 -->

<!-- You might want to consider using something like the [All Contributors](https://github.com/all-contributors/all-contributors) specification and its [emoji key](https://allcontributors.org/docs/en/emoji-key). -->

## Contact

If you want to contact me you can reach me at <m (dot) j (dot) carter (at) liverpool (dot) ac (dot) uk>.

## Citation
We appreciate citations as they let us discover what people have been doing with the software. 

To cite FilterPy in publications use:

Carter, M. (2023). FilterPy (1.0.0). https://github.com/mjcarter95/FilterPy

Or use the following BibTeX entry:

```
@misc{filterpy,
  title = {FilterPy (1.0.0)},
  author = {Carter, Matthew},
  year = {2023},
  month = May,
  howpublished = {GitHub},
  url = {https://github.com/mjcarter95/FilterPy}

}

