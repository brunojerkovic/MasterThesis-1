import numpy as np
from scipy import stats

def activation(z, text):
    """
    Calculate activation function.
    """
    if text == 'linear':
        return z
    elif text == 'sigmoid':
        return 1. / (1. + np.exp(-z))
    elif text == 'tanh':
        return np.tanh(z)
    elif text == 'relu':
        return np.max(0, z)

def sigmoid(z):
    """
    Calculate sigmoid function.
    """
    return 1. / (1. + np.exp(-z))

def log_ig(x, a, b):
    """
    Inverse Gamma distribution.
    """
    return a * np.log(b) - stats.gamma.pdf(a) - (a+1) * np.log(x) - b / x

def log_normal(x, mu, var):
    """
    Normal distribution.
    """
    return np.log(stats.norm.pdf(x, mu, np.sqrt(var)))

def log_beta(x, a, b):
    """
    Beta distribution.
    """
    return np.log(stats.beta.pdf(x, a, b))

def logpdf(theta, prior):
    """
    Calculate log-pdf of some distribution.
    """
    if prior[0] == 'normal':
        return log_normal(theta, prior[1][0], np.sqrt(prior[1][1]))
    elif prior[0] == 'ig':
        return log_ig(theta, prior[1][0], prior[1][1])
    elif prior[0] == 'beta':
        return log_beta(theta, prior[1][0], prior[1][1])

def log_jacobian(value, type):
    """
    Some log-jacobian handler. Note: value is the value of the original parameter, not the one from the random-walk proposal
    """
    if type == 'linear':
        return 0
    elif type == 'log':
        return np.log(value)
    elif type == 'logit':
        return np.log(value) + np.log(1-value)

def transform(value, type):
    """
    Transformation based on prior
    """
    if type == 'linear':
        return value
    elif type == 'log':
        return np.log(value)
    elif type == 'logit':
        return np.log(value / (1-value))

def inv_transform(value, type):
    """
    Inverse transformation after obtaining a parameter from a random walk proposal.
    """
    if type == 'linear':
        return value
    elif type == 'log':
        return np.exp(value)
    elif type == 'logit':
        return sigmoid(value)

def random_generator(dist):
    """
    Random number generator.
    """
    if dist.name == 'normal':
        return np.random.normal(dist.val[0], dist.val[1])
    elif dist.name == 'gamma':
        return "random('gam', dist.val(1), dist.val(2))"
    elif dist.name == 'ig':
        return "1 / random('gam', dist.val(1), dist.val(2))"
    elif dist.name == 'beta':
        temp = "betarnd(dist.val[0], dist.val[1])"
        return 2 * temp - 1

def update_scale(sigma2, acc, p, i, d):
    """
    Update scale factor of covariance matrix of the random-walk proposal.
    """
    T = 200
    alpha = "-norminv(p/2)"
    c = ((1-1/d) * np.sqrt(2*np.pi) * np.exp(alpha**2 / 2) / (2*alpha) + 1/(d*p*(1-p)))
    theta = np.log(np.sqrt(np.abs(sigma2)))
    theta = theta + c * (acc-p) / np.max(T, i/d)
    theta = np.exp(theta)
    theta = theta**2

    return theta

"""
function [B_var] = jitChol(B_var)
            % Cholesky decompostion
            [~,p] = chol(B_var);
            if p>0
                min_eig = min(eig(B_var));
                d       = size(B_var,1);
                delta   = max(0,-2*min_eig+10^(-5)).*eye(d);
                B_var   = B_var+delta;
            end
        end
"""

def rs_multinomial(w):
    """
    Binomial resampling.
    """
    N = len(w) # Number of particles
    idx = np.zeros(N) # Preallocate
    Q = np.cumsum(w) # Cumulative sum
    u = np.sort(np.random.uniform(size=N))
    j = 0
    for i in range(N):
        while (Q[j] < u[i]):
            j += 1 # Climb the ladder
        idx[i] = j # Assign index
    return idx

def rs_multinomial_sort(particles, w, u):
    """
    Binomial resampling with sorting.
    """
    N = len(w)  # Number of particles

    # MISLIN DA JE OVAJ KOD DOLJE SORTIRANJE SVEUKUPNO
    orig_index = np.arange(0, N, 1)
    cols = np.array([particles, w, orig_index]).T
    particles_sort = np.sort(cols[:, 0]) # TODO: sortrows[:,1]
    weight_sort = np.sort(cols[:, 1])
    orig_index_sort = np.sort(cols[:, 2]) # Rearange according to index of sorted particles

    indx_sort = np.zeros(N)  # Preallocate
    Q = np.cumsum(weight_sort)  # Cumulative sum
    Q[-1] = 1  # Make sure that the sum of weights is 1
    u = np.sort(u)

    j = 0
    for i in range(N):
        while Q[j] < u[i]:
            j += 1
        indx_sort[i] = j

    return indx_sort.astype(np.int32)

class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value




"""
        % Binomial resampling with sorting
        function indx = rs_multinomial_sort(particles,w,u)
            N = length(w);                       % Number of particles
            orig_index = (1:1:N);
            col = [particles',w',orig_index'];
            col_sort = sortrows(col,1);          % Hilbert sort  
            particles_sort = col_sort(:,1);  
            weight_sort = col_sort(:,2);         % Re-arrange weights according to idx of sorted particles
            orig_ind_sort = col_sort(:,3);
            indx_sort = zeros(1,N);              % Preallocate 
            Q = cumsum(weight_sort);             % Cumulative sum
            Q(end) = 1;                          % Make sure that sum of weights is 1
            u = sort(u);                         % Random numbers
            j = 1;
            for i = 1:N
                while (Q(j)<u(i))
                    j = j + 1;                   % Climb the ladder
                end
                indx_sort(i) = j;                % Assign index
            end
            indx = orig_ind_sort(indx_sort');
            indx = indx';
        end
"""

"""
        function indx = rs_multinomial_corr(w,u_res)
            N = length(w); % number of particles
            u_res = reshape(u_res,1,N);
            indx = zeros(1,N); % preallocate 
            Q = cumsum(w); % cumulative sum
            u = sort(u_res); % random numbers
            j = 1;
            for i=1:N
                while (Q(j)<u(i))
                    j = j+1; % climb the ladder
                end
                indx(i) = j; % assign index
            end
        end
"""

"""
        %% Forecast scores for Stochastic volatility models 
        function f = crps_normal(x,mu,sigma2)
            % Compute the predictive score (continuous ranked probability score - CRPS)
            % for normal distribution. The smaller CRPS the better prediction. 
            % See Gneiting, T., Raftery, A.: Strictly proper scoring rules, prediction, and
            % estimation. J. Am. Stat. Assoc. 102, 359–378 (2007)

            z = (x-mu)./sqrt(sigma2);
            f = sqrt(sigma2)*(1/sqrt(pi)-2*normpdf(z)-z.*(2*normcdf(z)-1));
        end

"""

"""
        function f = indicator_fun(y,quantile)
            if y<=quantile
                f = 1;
            else
                f = 0;
            end
        end
"""

"""
        %% Some helper function
        
        % Convert an array to a struct with each given field name for each
        % array element
        function theta_struct = array2struct(theta,name_list)
 
            % Make sure size is match
            theta = reshape(theta,1,length(theta));
            name_list = reshape(name_list,1,length(name_list));
            
            % Conver array to struct with given field names
            theta_struct = cell2struct(num2cell(theta),name_list,2);
        end
        
        function msg_out = errorMSG(identifier)

            switch identifier
                case 'error:DimensionalityPositiveInteger'
                    msg_out = 'Dimensionality must be integer';
                case 'error:DistributionNameMustBeString'
                    msg_out = 'Distribution Name must be string';
                case 'error:DistributionNameIsConstant'
                    msg_out = 'Distribution Name is constant';
                case 'error:NormalDistributionDimensionIncorrect'
                    msg_out = 'The parameter for a normal distribution should be a 1x2 array';
                case 'error:ModelMustBeSpecified'
                    msg_out = 'The fitted model and data must be specified';
                case 'error:DistributionMustBeBinomial'
                    msg_out = 'Binomial distribution option required';
                case 'error:MustSpecifyActivationFunction'
                    msg_out = 'Activation function type requied';
            end
        end
"""