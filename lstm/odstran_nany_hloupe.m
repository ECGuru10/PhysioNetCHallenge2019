
function [A,Anan]=odstran_nany_hloupe(A)
%úplnì chybìjící sou nahrazené prùmìrama z ostatních pacošù
%jiné lineární interpolací

mean_feature=nanmean(cat(1,A{:}),1);


for k=1:size(A,2)
    AA=A{k};
    Anan{k}=isnan(AA);
    for kk=1:size(AA,2)
%         if sum(isnan(AA(:,kk)),1)==numel(AA(:,kk))
%             AA(:,kk)=mean_feature(kk);
%         elseif sum(isnan(AA(:,kk)),1)==(numel(AA(:,kk))-1)
%             AA(:,kk)=fillmissing(AA(:,kk),'nearest');%pokud je jen jedno cislo tak linear nefunguje
%         else
%             AA(:,kk)=fillmissing(AA(:,kk),'linear');
% 
%         end
        AA(:,kk)=fillmissing(AA(:,kk),'previous');
        
        AA(isnan(AA(:,kk)),kk)=mean_feature(kk);

    end
    
    A{k}=AA;
end


end
