## Supplementary code for Neurips submission 7410

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

    
