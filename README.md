
# Differentiable Quantum Computing for Large-scale Linear Control (NeurIPS 2024)

[Connor Clayton*](https://quics.umd.edu/people/connor-clayton), [Jiaqi Leng*](https://jiaqileng.github.io/), [Gengzhi Yang*](https://qcrypt2021.quics.umd.edu/people/gengzhi-yang), [Yi-Ling Qiao](https://ylqiao.net/), [Ming C. Lin](https://www.cs.umd.edu/~lin/), [Xiaodi Wu](https://www.cs.umd.edu/~xwu/)

 [[NeurIPS]](https://nips.cc/virtual/2024/poster/95915) [[GitHub]](https://github.com/YilingQiao/diff_lqr)


## Demos

1. To optimize the control policy `K`. Just run 

```
python lqr2.py
```

2. We can switch between the classical method and ours by setting
`name = 'classical'` or `name = 'quantum'` in `lqr2.py`. Logs are stored in `./logs`

3. To run the scaling comparison, run 
```
python scale_comparison.py
```

## BibTex
```
@inproceedings{
clayton2024differentiable,
title={Differentiable Quantum Computing for Large-scale Linear Control},
author={Clayton, Connor and Leng, Jiaqi and Yang, Gengzhi and Qiao, Yi-Ling and Lin, Ming and Wu, Xiaodi},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
}
```
