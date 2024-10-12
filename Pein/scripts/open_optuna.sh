#!/bin/bash

# Set the tmux session name
TMUX_SESSION="dashboard"

# Set the MySQL URL for Optuna
SQL_URL="mysql://root:@localhost:3306/optuna_db"

# Set the port for the Optuna dashboard
OPTUNA_PORT=8081

# Check if the tmux session exists, if not create it
tmux has-session -t $TMUX_SESSION 2>/dev/null

if [ $? != 0 ]; then
    tmux new-session -d -s $TMUX_SESSION
fi

# Attach to the tmux session and run the Optuna dashboard command with the specified port
tmux send-keys -t $TMUX_SESSION "optuna-dashboard $SQL_URL --port $OPTUNA_PORT" C-m

# Attach to the tmux session
tmux attach-session -t $TMUX_SESSION
