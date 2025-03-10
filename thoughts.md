## formula:

The variables `u1p` and `u1pp` refer to the previous step of the simulation, and the previous previous step of the simulation, respectively. So if `u1` is at `t=10` then `u1p`  is at `t=9` and `u1pp` is at `t=8`. Same with `u2` and `u3`. 

```
Ux_nxt[x, y, z] = (dt**2*((lambda[x, y, z] + mu[x, y, z])*(Uy[x - 1, y - 1, z] - Uy[x - 1, y + 1, z] - Uy[x + 1, y - 1, z] + Uy[x + 1, y + 1, z]) + (lambda[x, y, z] + mu[x, y, z])*(Uz[x - 1, y, z - 1] - Uz[x - 1, y, z + 1] - Uz[x + 1, y, z - 1] + Uz[x + 1, y, z + 1]) + 4*(lambda[x, y, z] + 2*mu[x, y, z])*(-2*Ux[x, y, z] + Ux[x - 1, y, z] + Ux[x + 1, y, z]) + 4*(-2*Ux[x, y, z] + Ux[x, y, z - 1] + Ux[x, y, z + 1])*mu[x, y, z] + 4*(-2*Ux[x, y, z] + Ux[x, y - 1, z] + Ux[x, y + 1, z])*mu[x, y, z])/4 + (2*Ux[x, y, z] - Ux_prv[x, y, z])*rho[x, y, z])/rho[x, y, z];

Uy_nxt[x, y, z] = (dt**2*((lambda[x, y, z] + mu[x, y, z])*(Ux[x - 1, y - 1, z] - Ux[x - 1, y + 1, z] - Ux[x + 1, y - 1, z] + Ux[x + 1, y + 1, z]) + (lambda[x, y, z] + mu[x, y, z])*(Uz[x, y - 1, z - 1] - Uz[x, y - 1, z + 1] - Uz[x, y + 1, z - 1] + Uz[x, y + 1, z + 1]) + 4*(lambda[x, y, z] + 2*mu[x, y, z])*(-2*Uy[x, y, z] + Uy[x, y - 1, z] + Uy[x, y + 1, z]) + 4*(-2*Uy[x, y, z] + Uy[x, y, z - 1] + Uy[x, y, z + 1])*mu[x, y, z] + 4*(-2*Uy[x, y, z] + Uy[x - 1, y, z] + Uy[x + 1, y, z])*mu[x, y, z])/4 + (2*Uy[x, y, z] - Uy_prv[x, y, z])*rho[x, y, z])/rho[x, y, z];

Uz_nxt[x, y, z] = (dt**2*((lambda[x, y, z] + mu[x, y, z])*(Ux[x - 1, y, z - 1] - Ux[x - 1, y, z + 1] - Ux[x + 1, y, z - 1] + Ux[x + 1, y, z + 1]) + (lambda[x, y, z] + mu[x, y, z])*(Uy[x, y - 1, z - 1] - Uy[x, y - 1, z + 1] - Uy[x, y + 1, z - 1] + Uy[x, y + 1, z + 1]) + 4*(lambda[x, y, z] + 2*mu[x, y, z])*(-2*Uz[x, y, z] + Uz[x, y, z - 1] + Uz[x, y, z + 1]) + 4*(-2*Uz[x, y, z] + Uz[x, y - 1, z] + Uz[x, y + 1, z])*mu[x, y, z] + 4*(-2*Uz[x, y, z] + Uz[x - 1, y, z] + Uz[x + 1, y, z])*mu[x, y, z])/4 + (2*Uz[x, y, z] - Uz_prv[x, y, z])*rho[x, y, z])/rho[x, y, z];
```
## basic concept

this formula computes the next step from 3 previous steps.

- What is my initial situation? -> wave started from sender
- should I compute the formula for the whole space at the same time, then triple for loop over whole model ?
- until when do we compute ? 
- do I need 2 steps to start the simulation correctly ? (uprev and u)
- wait why does the formula use the current value of other dims ? I don't think I can use the current time neighbors to determine myself

- try random value in middle to test

- export data: export all frames (or 1 slice and visualize it)

- 


## actual use of the formula

I don't want to store a whole matrix for lambda, mu and rho, but I can make functions that compute their value at this position, looking at the model.

Then I keep in memory u1, u2, u3, and their p and pp version. => 9 full matrix, which would be quite big (model is 300x400x1200 or smth)

## implementation

what spatial scale do we use ? I guess indices for the whole space => so dx = 1, but we can change this later.

ok so the model seems to be stored as a 2d matrix of the height of the thing. then I only need to know 

Then i can just check on the x-y dimension if my current z (depth) coordinate is in the model or not

So what do I need ? I need the height of the plastic model. and to determine what I have where


### code structure

- init memory
- time step
- output data (what do we want to see ?)
- destroy memory

->basically just copy the 2d example, in 3d, with updated formula

### optimization

I can replace the model values by  the values of an enum, it would take less space and easier to understand

### upgrades

- add air over water,and take into account in simulation since wave can go out and come back