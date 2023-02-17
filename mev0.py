import numpy as np 
from decimal import localcontext, Decimal, ROUND_HALF_UP

# original documentation, still mostly relevant, though not entirely
"""
sim2.m runs the Markov chain Monte-Carlo version of the
algorithm. This version runs a number of iterations of a sampling
algorithm, whose distribution gradually converges to the output
distribution of the system. How long it takes depends on the settings
of Niters - the larger this variable is set, the longer it takes, and
the more accurately the result distribution will be to the desired
distribution. Niters=1000 seems a reasonable compromise.


Usage: [outorder, outorderpos, outorderlist, outorderprobs] = sim2(Niters, Nc, Nv, Nvg, Nparticles, marginstableorcreateseed, inorders);

Function to simulate the maximum entropy election system 
 (without voter ability to conjoin candidates) 
 by MCMC and adaptation of Lagrange multipliers.
If marginstableorcreateseed is 1x1 or empty then a set of votes 
 is created and its analysis simulated; if 1x1 then that value is 
 used to seed the random number generators used for creation;
 in this case Nc and Nv are respectively the number
 of candidates and the number of voters, taken to be 3 and 50
 if not given, and Nvg is the number of different types of voter,
 taken to be 3 if not given.
If marginstableorcreateseed is N x N then Nc is taken to be N and 
 the passed values of Nc and Nv are ignored; the matrix is 
 taken to be the marginstable C, such that C(nc1, nc2) is the 
 number of voters that thought candidate nc1 was better
 than or equal to candidate nc2.
If marginstableorcreateseed is Nv x Nc or inadmissible as a 
 (unnormalised) marginstable then it is taken to be a scores
 table; each voter in it will be represented by 100 voters with
 orderings derived by taking Gaussianly distributed scores around
 the given values.
If inorders is passed, of size Nc x Nparticles, and we are not
 synthesising a problem, then those orders are used to initialise 
 the particles, and Nparticles is set from this parameter; 
 inorders should contain randomly drawn orders from the votes.
 If we are synthesising a problem then inorders is ignored.
Nparticles is the number of Markov chains that are run; each 
 starts from the votes of a single randomly chosen voter (if available)
 or from a randomly chosen ordering otherwise.
outorder is a Nc x 1 vector which gives the output ordering
 of the candidates starting with the most preferred.
outorderpos is a Nc x 1 vector such that outorderpos(nc) is the 
 position in outorder of candidate nc.
outorderlist is a Nc x Nparticles x Niters list of all the samples
 considered.
outorderprobs is a Niters x 1 list of the probabilities with which
 each niter in outorderlist was considered for being outorder;
 the probability for each particle for any given iter is identical.
"""
def sim2(
        Niters = 1000, 
        Nc = 3,             # number of candidates
        Nv = 50,            # number of voters
        Nvg = None,         # number of voter groups
        Nparticles = 100, 
        marginstableorcreateseed = None, 
        inorders = None, 
        debug_print = True,
        report_debug_interval = 100,
        seed = 0,
        ): 

    np.random.seed(seed)
    eps = np.finfo(np.float32).eps  # for numerical stability purposes

    if Nvg is None:
        Nvg = Nc

    if inorders: 
        Nparticles = inorders.shape[1] 

    debug_print and print(f'Niters is {Niters}')

    if marginstableorcreateseed is None or np.isscalar(marginstableorcreateseed) or np.prod(marginstableorcreateseed.shape) <= 1: 
        synthesising = True
    else: 
        C = marginstableorcreateseed    # Nc x Nc margins table or Nv x Nc scores table
        synthesising = False 
    
    marginsnotscores = False
    if not synthesising: 
        # We need to sort out whether the table is a scores table or a margins table.
        if C.shape[0] == C.shape[1]: 
            if (abs(1 - (C + C.T) / (2 * C[0, 0])) < 1e-6).all():
                marginsnotscores = True
                Nc = C.shape[0]
        if not marginsnotscores: 
            Nv, Nc = C.shape
            C = C.T 

    if marginsnotscores:
        debug_print and print('Input is being taken as margin fractions rather than scores.')
    else:
        debug_print and print('Input is being taken as scores not margin fractions.')
    
    # Set various parameters for the analysis.

    # Number of candidates to aim to move each time.
    Ncdraw = 8

    # Minimum limit for averaging length
    minavlen = np.ceil(100 / Nparticles)

    # Amount to nudge betas.
    epsilon = 0.1

    # Do the synthesising if necessary
    if synthesising: 

        # First we must determine the score distributions for each candidate and each voter group
        scoremeans = np.random.rand(Nc, Nvg)

        scorestrengths = np.random.gamma(2, 1/0.1, (Nc, Nvg))

        # Then we assign a voter group to each voter,
        crits = np.tile(1 / Nvg, (Nvg, 1))
        crits = crits.cumsum(0)
        nvgs = np.minimum(Nvg-1, (np.tile(np.random.rand(1, Nv), (Nvg, 1)) > np.tile(crits, (1, Nv))).sum(0))

        # and propagate the means and strengths to each voter.
        scoremeans = scoremeans[:, nvgs]
        scorestrengths = scorestrengths[:, nvgs]

        # Then we assign a score to each voter for each candidate
        scores = np.zeros((Nc, Nv))
        for nc in range(Nc):
            for nv in range(Nv): 
                scores[nc, nv] = np.random.beta(scoremeans[nc, nv] * scorestrengths[nc, nv], \
                                        (1 - scoremeans[nc, nv]) * scorestrengths[nc, nv])

        marginsnotscores = False
        betascale = 10
        C = scores * betascale

    if not marginsnotscores:

        # Work out the margins table.
        scores = C
        Nvmult = 100
        scores = np.tile(scores, (1, Nvmult)) + np.random.randn(Nc, Nv * Nvmult)
        Nv = Nv * Nvmult

        # Then we work out the margins table
        C = np.zeros((Nc, Nc))
        for nc1 in range(Nc): 
            for nc2 in range(Nc): 
                C[nc1, nc2] = (scores[nc1, :] > scores[nc2, :]).sum() \
                            + 0.5 * (scores[nc1, :] == scores[nc2, :]).sum()

    # Check that C has the desired properties
    assert len(C.shape) == 2, 'marginstable is not 2-dimensional'
    assert C.shape[0] == Nc and C.shape[1] == Nc, 'marginstable is not Nc x Nc'
    Ccheck = C + C.T
    Nv = Ccheck.mean()
    Ccheck = Ccheck / Nv
    assert (Ccheck > 0.9999).all() and (Ccheck < 1.0001).all(), 'marginstable is not valid'

    C = C / Nv
    marginfracs = C

    # We've now finished synthesising and checking everything, so time to turn to analysis.

    if not marginsnotscores: 
        # Then we draw inorders.
        inordervoters = np.floor(Nv * np.random.rand(Nparticles)).astype(int)  # floor instead of ceil because of 0 vs 1 indexing 
        inorders = np.tile(np.nan, (Nc, Nparticles))
        for nparticle in range(Nparticles):
            ind = np.argsort(- scores[:, inordervoters[nparticle]])
            inorders[:, nparticle] = ind
        inorders = inorders.astype(int)

    # beta(nc1, nc2) is the number of nepers of favour to give to orderings that prefer nc1 to nc2.
    if False:
        # Initialise the betas at zero.
        betas = np.zeros((Nc, Nc))
    else: 
        cliplimit = 1e-4
        betas = 0.5 * np.log(np.maximum(marginfracs, cliplimit) / np.maximum(marginfracs.T, cliplimit))
    
    # Initialise the output ordering.
    if inorders is not None and inorders.size != 0: 
        outorders = inorders
    else: 
        outorders = np.tile(np.nan, (Nc, Nparticles))
        for nparticle in range(Nparticles): 
            outorders[:, nparticle] = np.random.permutation(Nc).T 
    outorders = outorders.astype(int)

    # Record variables.
    betaslist = np.tile(np.nan, (Nc, Nc, Niters))
    acceptlist = np.tile(np.nan, (Nparticles, Niters))
    outorderlist = np.tile(np.nan, (Nc, Nparticles, Niters))
    outorderposlist = np.tile(np.nan, (Nc, Nparticles, Niters))
    epsilonlist = np.tile(np.nan, (Niters, 1))
    rmserrorslist = np.tile(np.nan, (Niters, 1))
    decayfactorlist = np.tile(np.nan, (Niters, 1))

    # Iterate until converged.
    niter = 0
    useIIRcurrentmargins = 1
    if useIIRcurrentmargins:
        currentmargins = marginfracs
    else:
        currentmargins = np.tile(0.5, (Nc, Nc))

    while niter < Niters: 
        niter = niter + 1

        # First we nudge the betas.

        # We calculate how far to nudge them based on how far out we are with each marginfrac.

        if False: # *****
            nudgescale = np.abs(currentmargins - marginfracs)
        else: 
            nudgescale = np.abs(currentmargins - marginfracs) / np.maximum(1e-3, np.minimum(marginfracs, 1 - marginfracs)) 
   
        outorderposs = np.zeros(Nc*Nparticles)
        sub2ind = np.vectorize(lambda x,y: np.ravel_multi_index((x, y), dims=(Nc, Nparticles), order='F'))  # F for Fortran-style arrays (1 indexed / column-major)
        idx = sub2ind(outorders.ravel('F'), np.tile(np.arange(Nparticles, dtype=int), (Nc, 1)).ravel('F')) 
        outorderposs[idx] = np.tile(np.arange(Nc).reshape(Nc, 1), (1, Nparticles)).ravel('F')
        outorderposs = outorderposs.reshape(Nparticles, Nc).T.astype(int)
        thesecomparisons = np.zeros((Nc, Nc, Nparticles))

        for nparticle in range(Nparticles): 
            outorderpos = outorderposs[:, nparticle]
            nn1, nn2 = np.meshgrid(outorderpos, outorderpos, indexing='ij')
            thiscomparison = (nn1 < nn2) + 0.5 * (nn1 == nn2) # Set to 1 if nn1 better than nn2 or 0 otherwise.
            thesecomparisons[:, :, nparticle] = thiscomparison
        thiscomparison = thesecomparisons.sum(2) / Nparticles

    # **** question is whether this should have
    # **** nudgescale = 1 ./ max(1e-3, min(marginfracs, 1 - marginfracs)); and
    # **** update delta = sqrt(Nparticles * epsilon * nudgescale) .* sqrt(abs(thiscomparison - marginfracs)) .* sign(thiscomparison - marginfracs);;

        betas = betas - np.sqrt(Nparticles * epsilon * np.abs(thiscomparison - marginfracs) * nudgescale) * np.sign(thiscomparison - marginfracs)
        betas = (betas - betas.T) / 2

        Eolds = (-(thesecomparisons * np.tile(np.expand_dims(betas, -1), (1, 1, Nparticles))).sum((0, 1))).reshape(Nparticles, 1)

        # Redraw outorders, by Metropolis-Hastings, using the proposal distribution outlined in testhypothesis.m .

        for nparticle in range(Nparticles):
            outorder = outorders[:, nparticle]
            outorderpos = outorderposs[:, nparticle]
        
            # We will only redraw certain of the candidates.
            moving = np.random.rand(Nc, 1) < (Ncdraw / Nc)
            posmoving = np.where(moving)[0]
            ncsmoving = outorder[posmoving]
        
            # We first calculate the contributions to the proposal dist.
            Econtribs = -betas[ncsmoving, :]
            Econtribs = Econtribs + np.log(np.exp(Econtribs) + np.exp(-Econtribs))
        
            # Make a copy for updating in due course to calculate probability of backward move.
            Eoldcontribs = np.copy(Econtribs)
            oldmoving = np.copy(moving)
        
            # We start by considering what's going in the top moving position, and work downwards.
            neworder = outorder
            Eforward = 0
            Ebackward = 0
            for nposmoving in range(posmoving.shape[0]): 
        
                neworderpos = np.zeros((Nc, 1), dtype=int)
                neworderpos[neworder] = np.expand_dims(np.arange(Nc, dtype=int), -1)
        
                ncsbelow = np.where(neworderpos >= posmoving[nposmoving])[0]
                oldncsbelow = np.where(outorderpos >= posmoving[nposmoving])[0]
        
                Etot = Econtribs[:, ncsbelow].sum(1)
                Etot[~moving[posmoving].flatten()] = np.inf  # Kill those that are not up for moving.
                Etotmin = Etot.min()
                Etot = Etot - Etotmin if Etotmin != np.inf else Etot   # this prevents errors when Etot is all np.inf
                ptot = np.exp(-Etot)

                ptotsum = ptot.sum() 
                ptot = ptot / ptotsum if ptotsum != 0 else ptot + 1/ptot.shape[0]  # likewise
        
                chosennposmoving = np.random.choice(ptot.shape[0], p=ptot[:])
                Eforward = Eforward - np.log(ptot[chosennposmoving])
        
                Eoldtot = Econtribs[:, oldncsbelow].sum(1)
                Eoldtot[~oldmoving[posmoving].flatten()] = np.inf
                Eoldtotmin = Eoldtot.min() 
                Eoldtot = Eoldtot - Eoldtotmin if Eoldtotmin != np.inf else Eoldtot
                poldtot = np.exp(-Eoldtot)
                sumpoldtot = poldtot.sum()
                poldtot = poldtot / sumpoldtot 
                Eoldtot = Eoldtot + np.log(sumpoldtot) 

                Ebackward = Ebackward + Eoldtot[nposmoving]
        
                neworder[neworderpos[ncsmoving[chosennposmoving]]] = neworder[posmoving[nposmoving]]
                neworder[posmoving[nposmoving]] = ncsmoving[chosennposmoving]
                moving[posmoving[chosennposmoving]] = 0
                oldmoving[posmoving[nposmoving]] = 0
        
            neworderpos = np.zeros((Nc, 1))
            neworderpos[neworder] = np.expand_dims(np.arange(Nc), -1)
        
            nn1, nn2 = np.meshgrid(neworderpos, neworderpos, indexing='ij')
            newcomparison = (nn1 < nn2)
            Enew = -(newcomparison * betas).sum()
        
            # Now need to decide accept or reject ?
            Eaccept = Enew - Eolds[nparticle] + Ebackward - Eforward
            accept = np.random.rand() < np.exp(-Eaccept)
        
            if accept:
                outorder = np.copy(neworder)
                outorderpos = np.copy(neworderpos)
                outorders[:, nparticle] = outorder
                outorderposs[:, nparticle] = outorderpos.flatten()
            
            # Make a record for looking at later.
            betaslist[:, :, niter-1] = betas
            acceptlist[nparticle, niter-1] = accept
            outorderlist[:, nparticle, niter-1] = outorder
            outorderposlist[:, nparticle, niter-1] = outorderpos.flatten()

        # wantedavlen = max(minavlen, 2 / (Nparticles * epsilon ** 2))
        wantedavlen = max(minavlen, 2 / (Nparticles * max(epsilon ** 2, eps)))
        # python rounds halves down: https://docs.python.org/dev/library/functions.html#round
        # matlab/octave rounds halves up, so need to set up specific ROUND_HAL_UP enabled context here
        with localcontext() as ctx:
            ctx.rounding = ROUND_HALF_UP
            startav = round(max((Decimal(niter) / 4).to_integral_value(), niter - wantedavlen))
        avlength = niter - startav + 1

        if useIIRcurrentmargins:
            updatetimeconstant = max(minavlen, avlength)
            updatealpha = 1 / updatetimeconstant
            currentmargins = (1 - updatealpha) * currentmargins
            decayfactorlist[niter-1] = 1 - updatealpha
            for nc1 in range(Nc):
                for nc2 in range(Nc): 
                    currentmargins[nc1, nc2] = currentmargins[nc1, nc2] \
                                            + updatealpha * (outorderposlist[nc1, :, niter-1] < outorderposlist[nc2, :, niter-1]).sum() / Nparticles
            np.fill_diagonal(currentmargins, 0.5)
        else: 
            if niter % 100 == 0 or niter == Niters: 
                currentmargins = np.tile(np.nan, [Nc, Nc])
                for nc1 in range(Nc): 
                    for nc2 in range(Nc): 
                        currentmargins[nc1, nc2] = (outorderposlist[nc1, :, startav : niter-1] < outorderposlist[nc2, :, startav : niter-1]).sum() \
                                                / (avlength * Nparticles)
                currentmargins = currentmargins + 0.5 * np.eye(Nc)
 
        # ratios = currentmargins / marginfracs
        # errors = currentmargins - marginfracs
        ratios = currentmargins / np.clip(marginfracs, eps, 1-eps)
        errors = currentmargins - np.clip(marginfracs, eps, 1-eps)

        rmserrors = np.sqrt(np.mean(np.power(errors, 2)))

        epsilon = rmserrors / np.sqrt(max(minavlen, niter))

        if report_debug_interval > 0 and (niter % report_debug_interval == 0 or niter == Niters): 
            print('============ start report ============')
            print('niter =', niter)

            # Prepare an interim report.
            acceptrate = acceptlist[:, 0 : niter].sum() / (niter * Nparticles)
            print('acceptrate =', acceptrate)

            if Nc <= 5: 
                print('marginfracs =\n', marginfracs)
                print('currentmargins =\n', currentmargins)
                print('ratios =\n', ratios)
                print('errors =\n', errors)
            else: 
                ratioerrs = [ratios.max(), ratios.min()]
                maxerrors = np.abs(errors).max()
                print('ratioerrs =\n', ratioerrs)
                print('maxerrors =\n', maxerrors)

            print('rmserrors =', rmserrors)
            print('epsilon =', epsilon)
            print('============  end report  ============')

        # Record the various control variables.
        epsilonlist[niter-1] = epsilon
        rmserrorslist[niter-1] = rmserrors

    if useIIRcurrentmargins: 
        overalldecays = np.multiply(np.cumprod(np.append(decayfactorlist[1:], 1.)[::-1])[::-1], (1 - decayfactorlist).flatten())
        sumoveralldecays = overalldecays.sum()
        debug_print and print('sumoveralldecays =', sumoveralldecays)
        overalldecays = overalldecays / sumoveralldecays
        chosenone = np.random.choice(overalldecays.shape[0], p=overalldecays)
        outorderprobs = overalldecays
    else: 
        chosenone = int(np.random.rand() * avlength) + startav 
        outorderprobs = np.array([np.zeros(startav - 1, 1), np.tile(1 / avlength, [Niters - startav, 1])])
    
    chosenparticle = int(np.ceil(np.random.rand() * Nparticles))
    outorder = outorderlist.astype(int)[:, chosenparticle, chosenone]
    outorderpos = np.copy(outorder)
    outorderpos[outorder] = np.arange(Nc, dtype=int)

    return outorder, outorderpos, outorderlist, outorderprobs, chosenone, chosenparticle



if __name__=='__main__': 
    # outorder, outorderpos, outorderlist, outorderprobs, chosenone, chosenparticle = sim2(Niters=200, debug_print=False, report_debug_interval=0)

    marginstableorcreateseed = 10*np.random.rand(50, 3)
    outorder, outorderpos, outorderlist, outorderprobs, chosenone, chosenparticle = sim2(marginstableorcreateseed=marginstableorcreateseed,  Niters=200, debug_print=True, report_debug_interval=100)

    print()
    print(f'''Output:
        outorder: {outorder}
        outorderpos: {outorderpos}
        outorderprob: {outorderprobs[chosenone]}
    ''')
