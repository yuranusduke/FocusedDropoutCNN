# Unofficial Implementation of  ***FocusedDropout for Convolutional Neural Network***

Created by: Kunhong Yu (444447)/Islam Islamov ()/Leyla Ellazova ()

First of all, thank the authors very much for sharing this excellent paper [***FocusedDropout for Convolutional Neural Network***](https://arxiv.org/abs/2103.15425) with us. This repository contains FoucusedDropout and some basic implementation of experiments in paper. 
Specifically, we reproduce `Table 1`, `Table 2`, `Figure 4` and `Figure 5`. Since authors did not release original code and some important hyper-parameters, we try our best to achieve performance they claim in the paper, but the main motivation is to verify the proposed FocusedDropout is state-of-the-art, not to show exact testing accuracy, we use NVIDIA RTX A5000 to train and test all models, but we can not burden a lot of costs of all experiments, that's why we only reproduce parts of them.


## Contribution

In this part, we dynamically update our contribution in our group with three people.

- [x] 2023/04/07: Kunhong Yu creates original and empty repo to start project
- [x] 2023/04/08: Kunhong Yu add initial `config/cfg.yaml` file

## References
- FocusedDropout for Convolutional Neural Network [[ref]](https://arxiv.org/abs/2103.15425)


### To cite our work

```
@misc{FDI,
  author       = "KunhongYu/Islam Islamov/Leyla Ellazova",
  title        = "Unofficial implementation of FocusedDropout",
  howpublished = "\url{}",
  note         = "Accessed: 2023-xx-xx"
}
```

***<center>Veni，vidi，vici --Caesar</center>***
