This repo contains code to reproduce the paper: Neural sampling from cognitive maps enables goal-directed imagination and planning

Citation:
```bibtex
@article{lin2025neural,
  title={Neural sampling from cognitive maps supports goal-directed planning and imagination},
  author={Lin, Hui and Yang, Yukun and Zhao, Rong and Pezzulo, Giovanni and Maass, Wolfgang},
  journal={bioRxiv},
  pages={2025--05},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

### Running the notebooks

This project is setup with [pixi](https://pixi.prefix.dev/latest/).

```sh
# Install pixi
curl -fsSL https://pixi.sh/install.sh | sh

# The commands below require to move to this folder
pixi install

# if you only have CPU (e.g., MacOS)
pixi shell
# or, if you have an nvidia GPU
pixi shell -e gpu

jupyter notebook .
```
