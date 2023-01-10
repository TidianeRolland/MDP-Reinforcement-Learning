import numpy as ql


# The possible states when the agent is in a given state
# Get available states (you can go) from the current state
def get_possible_states(state):
    current_state_row = R[state,]
    states_possible = ql.where(current_state_row > 0)[1]
    return states_possible


def state_choice(available_states_range):
    # Si il y a au moins un state dans lequel on peut aller à partir de l'état courant
    if len(available_states_range) > 0:
        # Choisit un state au hasard parmi les states possibles
        next_state = int(ql.random.choice(available_states_range, 1))
    # Si aucun state n'est possible à partir de l'état courant
    else:
        # Choisit un state au hasard parmi tous les states
        next_state = int(ql.random.choice(5, 1))
    return next_state


def reward(state, next_state, const_gamma):
    max_value = ql.max(Q[next_state,])
    Q[state, next_state] = R[state, next_state] + const_gamma * max_value


def start(Q_val):
    # Liste des codes de chaque état
    codes_etats = ["A", "B", "C", "D", "E", "F"]
    map_etats = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    # Demande à l'utilisateur de saisir l'index de l'état de départ (A=0, B=1, C=2, etc.)
    etat_depart = input("Entrez l'état de départ (A, B, C, D, E, F) : ")
    print("Chemin de concepts :")
    print("->", etat_depart)

    # Initialisation de nextc et nextci
    index_suivant = -1
    index_arrivee = 2
    etat_actuel = map_etats[etat_depart]

    while etat_actuel != index_arrivee:
        index_suivant = ql.argmax(Q_val[etat_actuel, :])
        if index_suivant == etat_actuel:
            break
        print("->", codes_etats[int(index_suivant)])
        etat_actuel = index_suivant


R = ql.matrix([[0, 0, 0, 0, 1, 0],
               [0, 0, 0, 1, 0, 1],
               [0, 0, 100, 1, 0, 0],
               [0, 1, 1, 0, 1, 0],
               [1, 0, 0, 1, 0, 0],
               [0, 1, 0, 0, 0, 0]])
Q = ql.matrix(ql.zeros([6, 6]))
gamma = 0.8

for i in range(50000):
    current_state = ql.random.randint(0, int(Q.shape[0]))
    possible_states = get_possible_states(current_state)
    chosen_state = state_choice(possible_states)
    reward(current_state, chosen_state, gamma)

# Displaying Q before the norm of Q phase
print("Q :")
print(Q)
# Norm of Q
print("Normed Q :")
Q_norm = Q / ql.max(Q) * 100
print(Q_norm)

keep = ""
while keep != "S":
    start(Q_norm)
    keep = input("Entrez 'S' pour sortir, ou tapez n'importe quoi pour continuer...")
