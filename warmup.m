MyPath = userpath;
MyDir = MyPath(1:strfind(MyPath,';')-1);
MyWorkDir = genpath(MyDir);
addpath(MyWorkDir, '-end');
addpath(genpath(pwd));
clear
