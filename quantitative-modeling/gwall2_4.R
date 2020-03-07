#### Assignment 4 ####
#### Gordon Wall (gwall2) ####

#### Install lpSolveAPI package to facilitate solving our problem ####
install.packages("lpSolveAPI")

#### Access package library ####
library(lpSolveAPI)

#### Formulate LP Model ####
# create an lp object with 0 constraints and 9 decision variables
lpweigelt <- make.lp(0, 9)

# create a maximization objective function
set.objfn(lpweigelt, c(420, 360, 300, 420, 360, 300, 420, 360, 300))

lp.control(lpweigelt, sense='max')

# add constraints
add.constraint(lpweigelt, c(1, 1, 1, 0, 0, 0, 0, 0, 0), "<=", 750)
add.constraint(lpweigelt, c(0, 0, 0, 1, 1, 1, 0, 0, 0), "<=", 900)
add.constraint(lpweigelt, c(0, 0, 0, 0, 0, 0, 1, 1, 1), "<=", 450)

add.constraint(lpweigelt, c(20, 15, 12, 0, 0, 0, 0, 0, 0), "<=", 13000)
add.constraint(lpweigelt, c(0, 0, 0, 20, 15, 12, 0, 0, 0), "<=", 12000)
add.constraint(lpweigelt, c(0, 0, 0, 0, 0, 0, 20, 15, 12), "<=", 5000)

add.constraint(lpweigelt, c(1, 0, 0, 1, 0, 0, 1, 0, 0), "<=", 900)
add.constraint(lpweigelt, c(0, 1, 0, 0, 1, 0, 0, 1, 0), "<=", 1200)
add.constraint(lpweigelt, c(0, 0, 1, 0, 0, 1, 0, 0, 1), "<=", 750)

add.constraint(lpweigelt, c(1/750, 1/750, 1/750, -1/900, -1/900, -1/900, 0, 0, 0), "=", 0)
add.constraint(lpweigelt, c(0, 0, 0, 1/900, 1/900, 1/900, -1/450, -1/450, -1/450), "=", 0)

# default setting for variable bounds is >= 0, so no need to manually set bounds
# optional naming of decision variables and constraints
RowNames <- c("Plant1", "Plant2", "Plant3", "Plant1", "Plant2", "Plant3", "Plant1", "Plant2", "Plant3", "Net1", "Net2")
ColNames <- c("Plant1Lg", "Plant1Med", "Plant1Sm", "Plant2Lg", "Plant2Med", "Plant2Sm", "Plant3Lg", "Plant3Med", "Plant3Sm")
dimnames(lpweigelt) <- list(RowNames, ColNames)

# optional printout of formulated model
lpweigelt

# solving our formulated model (printout of the number zero indicates the solution was successfully determined)
solve(lpweigelt)

# printout of actual solution value and variable coefficient values
get.objective(lpweigelt)
get.variables(lpweigelt)

# solved!