%% For Lattice Simulations
% First we have to sort the "connected" table (1082 x 1) from the lowest values to the highest
sorted = sortrows(connected5);
sorted = table2array(sorted);

% Now we have to perform the powerlaw fit
r_plfit(sorted, 'fig')

% Now we have to extract the X data and Y data from the plot
% We have to do this manually because the function r_plfit does not return
% the X and Y data
X = get(gca,'Children');
X_data = get(X,'XData');
X_DATA_FIT = X_data{2, 1};
X_DATA_POINTS = X_data{3, 1};
Y = get(gca,'Children');
Y_data = get(Y,'YData');
Y_DATA_FIT = Y_data{2, 1};
Y_DATA_POINTS = Y_data{3, 1};

% Now we save the X and Y data as csv files with the connected prefix
csvwrite('connected_X_FIT_5.csv',X_DATA_FIT);
csvwrite('connected_X_POINTS_5.csv',X_DATA_POINTS);
csvwrite('connected_Y_FIT_5.csv',Y_DATA_FIT);
csvwrite('connected_Y_POINTS_5.csv',Y_DATA_POINTS);

%% For CPIM simulations

 % First we have to sort the "connected" table (1082 x 1) from the lowest values to the highest
sorted = sortrows(CPIMconnected5);
sorted = table2array(sorted);

% Now we have to perform the powerlaw fit
r_plfit(sorted, 'fig')

% Now we have to extract the X data and Y data from the plot
% We have to do this manually because the function r_plfit does not return
% the X and Y data
X = get(gca,'Children');
X_data = get(X,'XData');
X_DATA_FIT = X_data{2, 1};
X_DATA_POINTS = X_data{3, 1};
Y = get(gca,'Children');
Y_data = get(Y,'YData');
Y_DATA_FIT = Y_data{2, 1};
Y_DATA_POINTS = Y_data{3, 1};

% Now we save the X and Y data as csv files with the connected prefix
%csvwrite('connected_X_FIT_5.csv',X_DATA_FIT);
%csvwrite('connected_X_POINTS_5.csv',X_DATA_POINTS);
%csvwrite('connected_Y_FIT_5.csv',Y_DATA_FIT);
%csvwrite('connected_Y_POINTS_5.csv',Y_DATA_POINTS);