# ML Atlas

An interactive 3D learning experience for decision trees, empirical risk,
perceptrons, support-vector machines, kernels, regularization, and neural
networks.

[Open ML Atlas on GitHub Pages](https://ntz619.github.io/ml-atlas/)

## Open the single-file version

Download `standalone/index.html` and open it directly in a modern browser. It
contains the application code, styles, equations, and 3D scene in one file; no
server or internet connection is required.

To regenerate it after changing the source:

```powershell
npm install
npm run build:single
```

## Develop locally

```powershell
npm install
npm run dev
```

Open `http://localhost:3017`.

## Validation

```powershell
npm test
npm run lint
npm run build
```
