## formula:

The variables `u1p` and `u1pp` refer to the previous step of the simulation, and the previous previous step of the simulation, respectively. So if `u1` is at `t=10` then `u1p`  is at `t=9` and `u1pp` is at `t=8`. Same with `u2` and `u3`. 

```
u1[x, y, z] = (dt**2*((lambda[x, y, z] + mu[x, y, z])*(u2[x - 1, y - 1, z] - u2[x - 1, y + 1, z] - u2[x + 1, y - 1, z] + u2[x + 1, y + 1, z]) + (lambda[x, y, z] + mu[x, y, z])*(u3[x - 1, y, z - 1] - u3[x - 1, y, z + 1] - u3[x + 1, y, z - 1] + u3[x + 1, y, z + 1]) + 4*(lambda[x, y, z] + 2*mu[x, y, z])*(-2*u1[x, y, z] + u1[x - 1, y, z] + u1[x + 1, y, z]) + 4*(-2*u1[x, y, z] + u1[x, y, z - 1] + u1[x, y, z + 1])*mu[x, y, z] + 4*(-2*u1[x, y, z] + u1[x, y - 1, z] + u1[x, y + 1, z])*mu[x, y, z])/4 + (2*u1p[x, y, z] - u1pp[x, y, z])*rho[x, y, z])/rho[x, y, z];

u2[x, y, z] = (dt**2*((lambda[x, y, z] + mu[x, y, z])*(u1[x - 1, y - 1, z] - u1[x - 1, y + 1, z] - u1[x + 1, y - 1, z] + u1[x + 1, y + 1, z]) + (lambda[x, y, z] + mu[x, y, z])*(u3[x, y - 1, z - 1] - u3[x, y - 1, z + 1] - u3[x, y + 1, z - 1] + u3[x, y + 1, z + 1]) + 4*(lambda[x, y, z] + 2*mu[x, y, z])*(-2*u2[x, y, z] + u2[x, y - 1, z] + u2[x, y + 1, z]) + 4*(-2*u2[x, y, z] + u2[x, y, z - 1] + u2[x, y, z + 1])*mu[x, y, z] + 4*(-2*u2[x, y, z] + u2[x - 1, y, z] + u2[x + 1, y, z])*mu[x, y, z])/4 + (2*u2p[x, y, z] - u2pp[x, y, z])*rho[x, y, z])/rho[x, y, z];

u3[x, y, z] = (dt**2*((lambda[x, y, z] + mu[x, y, z])*(u1[x - 1, y, z - 1] - u1[x - 1, y, z + 1] - u1[x + 1, y, z - 1] + u1[x + 1, y, z + 1]) + (lambda[x, y, z] + mu[x, y, z])*(u2[x, y - 1, z - 1] - u2[x, y - 1, z + 1] - u2[x, y + 1, z - 1] + u2[x, y + 1, z + 1]) + 4*(lambda[x, y, z] + 2*mu[x, y, z])*(-2*u3[x, y, z] + u3[x, y, z - 1] + u3[x, y, z + 1]) + 4*(-2*u3[x, y, z] + u3[x, y - 1, z] + u3[x, y + 1, z])*mu[x, y, z] + 4*(-2*u3[x, y, z] + u3[x - 1, y, z] + u3[x + 1, y, z])*mu[x, y, z])/4 + (2*u3p[x, y, z] - u3pp[x, y, z])*rho[x, y, z])/rho[x, y, z];
```
## basic concept

this formula computes the next step from 3 previous steps.

- What is my initial situation? -> wave started from sender
- should I compute the formula for the whole space at the same time, then triple for loop over whole model ?
- until when do we compute ? 

## actual use of the formula

I don't want to store a whole matrix for lambda, mu and rho, but I can make functions that compute their value at this position, looking at the model.

Then I keep in memory u1, u2, u3, and their p and pp version. => 9 full matrix, which would be quite big (model is 300x400x1200 or smth)

## implementation

what spatial scale do we use ? I guess indices for the whole space => so dx = 1, but we can change this later.

### code structure

- init memory
- time step
- output data (what do we want to see ?)
- destroy memory

->basically just copy the 2d example, in 3d, with updated formula