import axelrod as axl

trainer = axl.DQN()

def play_with(opp, num):
    players = [opp, trainer]
    match = axl.Match(players, 200)
    trainer.receive_match_attributes()
    for i in range(num):
        match.play()
        print (trainer.replay_memory)

temp = axl.CyclerDC()
play_with(temp, 1)
temp = axl.TitForTats()
play_with(temp, 30)


#players = [axl.AntiTitForTat(),
#           axl.DQN_tester(),                   
#           axl.Alternator(),                    
#           axl.TitForTat(),                 
#           axl.Bully(),                         
#           axl.Cooperator(),                    
#           axl.CyclerDC(),                      
#           axl.Defector(),                      
#           axl.SuspiciousTitForTat()]            

#turns = 200
#tournament = axl.Tournament(players, turns=turns, repetitions=1, seed=75)
#results = tournament.play()
#for average_score_per_turn in results.payoff_matrix[-2]:
#    print(round(average_score_per_turn, 3))