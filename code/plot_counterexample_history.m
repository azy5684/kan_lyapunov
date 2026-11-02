clear; clc; close all;
code_dir = fileparts(mfilename('fullpath'));
output_dir = fullfile(fileparts(code_dir), 'figures');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
filename = fullfile(code_dir, 'figure_data', 'ce_history_boundary_adjustment.json');
history = jsondecode(fileread(filename));
assert(numel(history)==312, 'Expected original 312 counterexample records.');
epoch = [history.epoch]';
x1 = [history.x1]'; x2 = [history.x2]';
radius = hypot(x1,x2);
[epoch,order] = sort(epoch); radius = radius(order);
assert(min(epoch)==20 && max(epoch)==1480);
[steps,~,group] = unique(epoch);
nearest = accumarray(group,radius,[],@min);
farthest = accumarray(group,radius,[],@max);

% Target-radius schedule used by the figure.
schedule_epoch = [0,520,540,560,580,600,620,640,660,680,700,720,740,760,780,800,820,840,860,1540];
schedule_radius = [2,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.35];
phase_boundary = 520;

fig = figure('Color','w','Position',[100,100,700,530]);
ax = axes('Parent',fig,'Position',[.16,.15,.80,.81]); hold(ax,'on');
ax.Toolbar.Visible = 'off';
disableDefaultInteractivity(ax);
scatter(ax,epoch,radius,12,'MarkerFaceColor',[.5,.4,1], ...
    'MarkerEdgeColor',[.5,.4,1],'MarkerFaceAlpha',.26,'MarkerEdgeAlpha',.26, ...
    'DisplayName','CE points');
stairs(ax,schedule_epoch,schedule_radius,'Color',[.48,.48,.48], ...
    'LineWidth',1.4,'DisplayName','target radius');
plot(ax,steps,nearest,'Color',[1,.647,0],'LineWidth',1.6,'DisplayName','nearest CE');
plot(ax,steps,farthest,'Color',[.1,.6,.1],'LineWidth',1.6,'DisplayName','farthest CE');
set(ax,'FontName','Arial','FontSize',12,'LineWidth',1,'Box','on','TickDir','out');
xlim(ax,[0,max(epoch)]); ylim(ax,[0,2.5]);
xticks(ax,0:200:1400); yticks(ax,0:.5:2.5);
xlabel(ax,'iteration','FontSize',13);
ylabel(ax,'Distance of CE to the equilibrium point','FontSize',13);
legend(ax,'Location','northeast','FontSize',11,'Box','on');
grid(ax,'off');

drawnow;
pos=ax.Position;
x_to_fig=@(v) pos(1)+v/max(epoch)*pos(3);
y_to_fig=@(v) pos(2)+v/2.5*pos(4);
annotation(fig,'doublearrow',[x_to_fig(8),x_to_fig(phase_boundary-8)], ...
    [y_to_fig(.35),y_to_fig(.35)],'Color','r','LineStyle','--','LineWidth',1, ...
    'Head1Length',7,'Head2Length',7,'Head1Width',7,'Head2Width',7);
annotation(fig,'doublearrow',[x_to_fig(phase_boundary+8),x_to_fig(max(epoch)-8)], ...
    [y_to_fig(.35),y_to_fig(.35)],'Color','r','LineStyle','--','LineWidth',1, ...
    'Head1Length',7,'Head2Length',7,'Head1Width',7,'Head2Width',7);
plot(ax,[phase_boundary,phase_boundary],[.28,.48],'r-','LineWidth',1,'HandleVisibility','off');
text(ax,phase_boundary/2,.16,'Shrinking phase','Color','r', ...
    'HorizontalAlignment','center','FontName','Arial','FontSize',11);
text(ax,(phase_boundary+max(epoch))/2,.16,'Expanding phase','Color','r', ...
    'HorizontalAlignment','center','FontName','Arial','FontSize',11);

print(fig,fullfile(output_dir,'counterexample_distance.svg'),'-dsvg');
exportgraphics(fig,fullfile(output_dir,'counterexample_distance.png'),'Resolution',300);
fprintf('Plotted %d original records, epochs %d-%d.\n',numel(radius),min(epoch),max(epoch));
fprintf('Output: %s\n',output_dir);
