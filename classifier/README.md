# classifier

This folder contains the code that was used to classify (both automatically and manually) the LLM outputs into being 
supportive of Response A or Response B for each prompt and LLM.

- `classifier.py` contains the code aiming to assess whether the model has chosen
response A or response B.  
- `evaluation.py` contains the code that runs the predictor on the test data I
have to determine how good it is. (RUN THIS AS `__main__` TO GET STATS)  
- `manual.py` contains the code that enables manual classification of the
final few items that can't be classified automatically while saving
the state to allow manual classification over multiple sessions if desired.
(RUN THIS AS `__main__` TO DO MANUAL CLASSIFICATION AND GET OUTPUT CSVs)

So **to run the classifier and then get prompted for manual classification, run `python3 -m classifier.manual`**

In order to run the classifier for testing (i.e. having clean files already), set the constatants in `classifier.evaluation`. In order to apply it to unseen raw files, pass the list of files you want to classify to `classifier.manual` as command line arguments. For either case, probably want to set `EXCLUDE_COLUMNS_REGEX` in `classifier.evaluation`.

## Terminology

There are a lot of similar words for different things being used here so I
will define the terminology I have chosen for things like variable names very precisely:

- "Response {A,B}" - the two options given in the benchmark which the LLM
  has to decide between
- LLM Output - the output of the LLM including its explanation and things
  like that if relevant
- Class - the response (either A or B) that an LLM has given, decided either
  by human reviewers (manual) or by my automated system here (automatic)
