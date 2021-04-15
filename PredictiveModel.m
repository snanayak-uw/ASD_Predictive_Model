% Code is based on research done by:
% Noriaki Yahata, Jun Morimoto, Ryuichiro Hashimoto, Giuseppe Lisi, Kazuhisa Shibata, Yuki Kawakubo, Hitoshi Kuwabara,Miho Kuroda, Takashi Yamada, Fukuda Megumi, Hiroshi Imamizu, Jose E. Nanez, Hidehiko Takahashi, Yasumasa Okamoto, Kiyoto Kasai, Nobumasa Kato, Yuka Sasaki, Takeo Watanabe, and Mitsuo Kawato (2016).
% A Small Number of Abnormal Brain Connections Predicts Adult Autism Spectrum Disorder, NatComms

% Copyright (c) 2016, Giuseppe Lisi, ATR CNS, glisi@atr.jp

% Script written/modified by Sama Nanayakkara 
% Last modified: Apr. 15, 2021

%% input_info
    %%
    % Male patient info
    curr_fpath = pwd;
    input_path = [curr_fpath, '\M_info.mat'];
    input_mats = load(input_path);

    % connectivity matrices
    rest_1_mats = input_mats.M_stack;

    % behavioural data
    PMAT_CR = double(input_mats.M_DX');

    % confounding variables
    confound_path = [curr_fpath, '\M_confounds.mat'];
    confound_mat = load(confound_path);
    confound_mat.M_SITE = double(categorical(string(confound_mat.M_SITE)))';
    confounds = [double(confound_mat.M_DX)',double(confound_mat.M_SITE)',double(confound_mat.M_AGE)',double(confound_mat.M_SEX)',double(confound_mat.M_EYE)'];

    % randomly shuffle order of subjects
    no_sub = size(rest_1_mats,3); %number of subjects

    rng default
    idx = randperm(no_sub);
    rest_1_mats = rest_1_mats(:,:,idx);
    PMAT_CR = PMAT_CR(idx);
    confounds = confounds(idx,:);

    all_mats  = rest_1_mats; %MxMxN matrix of 2D connectivity matrices for N subjects
    all_behav = PMAT_CR; %Nx1 vector of ASD diagnosis (NT = 1, ASD = 2)
    all_confounds = confounds; %NxP matrix with P different confounding variables

    for i = 1:size(all_mats,3) %eliminate redundancy by keeping only the lower triangular portions
        all_mats(:,:,i) = tril(all_mats(:,:,i),-1);
    end

    all_vcts = reshape(all_mats,[],size(all_mats,3));
    all_vcts(all_vcts(:,1)==0,:) = [];

    % separate out a test set
    test_size = 30;
    ts = test_size - 1;

    test_vcts = all_vcts(:,end-ts:end);
    all_vcts(:,end-ts:end) = [];
    test_behav = all_behav(end-ts:end);
    all_behav(end-ts:end) = [];
    test_confd = all_confounds(end-ts:end,:);
    all_confounds(end-ts:end,:) = [];
    
    fprintf('\nData has been inputted.\n')
    fprintf('===============================================')

%% CCA_featureselection
    clearvars -except all_vcts all_behav all_confounds test_behav test_confd test_vcts %for clarity, reduces number of variables

    %% Initialization
    % initialize diagnostic matrix and site matrix for NV
    DX = all_confounds(:,1);
    no_sub = length(DX);
    DXmat = zeros(no_sub,2);

    sites = all_confounds(:,2);
    no_sites = length(unique(sites));
    siteMat = zeros(no_sub,no_sites);

    for i=1:no_sub
        DXmat(i,DX(i)) = 1;
        siteMat(i,sites(i)) = 1;
    end

    all_confounds = [DXmat siteMat all_confounds(:,3:end)];
    all_confounds(all_confounds == 0) = -1;

    % determine metrics
    n_conn = size(all_vcts,1);
    n_behav = size(all_confounds,2);
    n_sub = size(all_confounds, 1);
    n_can = size(all_confounds,2);

    % initialize lambda 
    lambda = 0.1:0.1:0.9;
    lam_combn = sortrows([lambda(:) lambda(:); nchoosek(lambda,2)]);
    n_lam = length(lam_combn);

    %% Outer loop feature selection
    no_sub = size(all_vcts,2); % number of subjects
    K = 9; % number of folds
    stratify_data = [DXmat siteMat]; % labels for diagnosis and site
    FS_parts = iterativeStratification(stratify_data,9); % partition number for each subject based on stratification data

    %check stratification
    divisions = zeros(K,n_behav);
    for i = 1:K
        divisions(i,:) = sum(all_confounds(FS_parts == i,:));
    end
    bar(divisions')

    all_idx = 1:no_sub; % list of all subject indices

    all_ConnCoeff = zeros(n_conn,n_behav, n_lam, K-1); %initialize weights of features
    all_NVcoeff = zeros(n_behav,n_behav,n_lam,K-1);

    for i = 1:K-1 % first 8 folds
        part_idx = find(FS_parts == i); % indices of fold
        fold_conn = all_vcts(:,part_idx)'; % n x p matrix of ROI-to-ROI connections for fold subjects
        fold_NV = all_confounds(part_idx,:);

        % Inner loop feature selection
        % L1-SCCA
        for j = 1:n_lam
            [all_ConnCoeff(:,:,j,i), all_NVcoeff(:,:,j,i)] = L1_SCCA(fold_conn,fold_NV,lam_combn(j,1),lam_combn(j,2),n_can);
        end

    end
    
    fprintf('\nL1-SCCA complete.\n')
    
    %% Determine which features were selected
    vD = [];
    for i = 1:K-1
        for j = 1:n_lam
            selcol = []; %column indices that correspond with diagnostical canonical constraint
            curr_ConnCoeff = all_ConnCoeff(:,:,j,i);
            curr_NVcoeff = all_NVcoeff(:,:,j,i);
            DX_rows = curr_NVcoeff(1:2,:);
            [~,NVcol] = find(DX_rows ~= 0);
            NVcol = unique(NVcol);
            for c = 1:length(NVcol) %for each nuisance variable
                if any(curr_NVcoeff(3:end,NVcol(c))) == 0 %if there is no correlation with variable other than diagnosis
                    selcol = [selcol NVcol(c)]; %keep the corresponding connection
                end
            end

            if ~isempty(selcol)
                Conn_col = curr_ConnCoeff(:,selcol);
                sum_Conn = sum(abs(Conn_col),2);
                vD = [vD sum_Conn];
            end

        end 
    end

    vD_union = sum(vD,2);
    sel_Conn = find(vD_union > 0); %indices of relevant connections
    
    fprintf('\nFeatures selection complete.\n')
    fprintf('===============================================\n')

%% LOO_crossvalidation
    %%
    sel_vcts = all_vcts(sel_Conn,:); %connectivity matrix with only selected features
    CV_idx = find(FS_parts == K);
    no_CV = length(CV_idx);
    tr_pred = zeros(no_CV, 1); %prediction accuracy log
    te_pred = zeros(no_CV, 1);
    ww_f = zeros(length(sel_Conn)+1,no_CV);

    errTable_tr = [];
    errTable_te = [];

    %% LOOCV
    for sub = 1:no_CV
        val_vcts = sel_vcts(:,CV_idx(sub))'; %connectivity of validation subject
        val_dx = DX(CV_idx(sub)); %diagnosis of validation subject
        tr_vcts = sel_vcts'; 
        tr_vcts(CV_idx(sub),:) = []; %remove validation subject from training set
        tr_dx = DX;
        tr_dx(CV_idx(sub)) = []; %remove validation diagnosis from training set

        % SLR
    %     [sub_pred] = slr(tr_vcts, tr_dx);
        [ww_f(:,sub), ix_eff_f, errTable_tr(:,:,sub), errTable_te(:,:,sub)] = biclsfy_slrvar(tr_vcts, tr_dx, val_vcts, val_dx,...
        'nlearn', 300, 'mean_mode', 'none', 'scale_mode', 'none', 'invhessian',0);
        tr_pred(sub) = calc_percor(errTable_tr(:,:,sub)); %track accuracy
        te_pred(sub) = calc_percor(errTable_te(:,:,sub));

    end  

    av_w = sum(ww_f,2);
    
    fprintf('\nCross-validation complete.\n')
    fprintf('===============================================')

    fprintf('\nTraining Accuracy : %.2f%%\n',mean(tr_pred))
    fprintf('\nValidation Accuracy : %.2f%%\n',mean(te_pred))

    te_vcts = [ones(size(test_vcts,2),1) test_vcts(sel_Conn,:)'];
    p = round(predict_log(av_w, te_vcts))+1;
    fprintf('\nTest Accuracy : %.2f%%\n',mean(p == test_behav)*100)