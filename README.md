# Conjugate Gradient Solver for sparse symmetric matrices 

<!-- ![image info](./media/cylinder.png =100x100) -->
<p align="center">
<img src="media/cylinder.png" width="400">
</p>

```text
conjugate_gradient/
├── include/        # Header files (.h)
│   ├── cg.h, coo.h, csr.h, ...
├── src/            # algorithm and matrix formats support (.c)
│   ├── cg.c, coo.c, csr.c, ...
├── tests/          
│   ├── tests_cg.c, test_matrix.c, ...  # tests (.c)
├── build/          # Executables 
├── data/           # OpenFoam i/o files
├── riscv/          # Riscv simulated experimets (Spike)
├── Makefile        
└── README.md       
```
Run CG using a sample problem from OpenFOAM:
```
make test_cg
./build/test_cg  
```

Supported matrix formats: 
```text
- [x] COO
- [x] CSR
- [x] ELL 
- [] SELL
```

