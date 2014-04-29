function [prob1, prob2, prob3] = ps1(training, test)
	D = load(training);
	T = load(test);
	[predicted_labels, accuracy, decisions] = p1(D,T);	
	prob1 = {predicted_labels, accuracy, decisions};
	prob2 = p2(T.test.labels, predicted_labels);
	%prob3 = p3(D.train{end}.images, D.train{end}.labels);
end

function [predicted_labels, accuracy, decisions] = p1(training_data, test_data)
	train_features = [];
	train_labels = [];
	predicted_labels = [];
	accuracy = [];
	decisions = [];
	test_features =  [];
	for y = 1:size(test_data.test.images,3)
               	test_features = [test_features, reshape(test_data.test.images(:,:,y),1,[])'];
	end

	test_labels =  double(test_data.test.labels);
	test_features = sparse(double(test_features));

	for i = 1: size(training_data.train,2)
		train_labels= double(training_data.train{1,i}.labels);
		train_features = [];
		for j = 1:size(training_data.train{i}.images,3)		    
			v = reshape(training_data.train{i}.images(:,:,j),1,[]);
			train_features = [train_features, v'];
		end 
		train_features = double(train_features);

		model = train(train_labels, sparse(train_features'), '-q');

		[predicted_label_subset, acc, d] = predict(test_labels, test_features', model, '-q');

		predicted_labels = [predicted_labels, predicted_label_subset];
		accuracy = [accuracy, acc];
		decisions = [decisions, d];  

	end

	err = [100 - accuracy(1,:)];
	trainingSizes = [];
	for i = 1:size(training_data.train,2)
		trainingSizes = [trainingSizes, size(training_data.train{1,i}.images,3)];
	end
	figure;
	plot (trainingSizes, err)	      
	title('Error Rate of SVM');
	xlabel('Training Set Size');
	ylabel('% Error');


end

function [confusion_matrices] = p2(predictions, actual)

	train_set_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]; 	
	confusionMat = [double(actual), double(predictions)];
	for i = 2:size(confusionMat, 2)
		figure;
		imagesc(confusionmat(confusionMat(:,1), confusionMat(:,i)))
		set(gca, 'XTick', [0:10]);
		set(gca, 'XTickLabel', [-1:9]);
		set(gca, 'YTick', [0:10]);
		set(gca, 'YTickLabel', [-1:9]);
		title(['Confusion Matrix of Model Trained on ', int2str(train_set_sizes(i-1)), 'examples']);
		xlabel('Predicted Label');
		ylabel('True Label'); 
		colorbar
        end
	confusion_matrices = confusionMat;
end


function [best] = p3(train_data,train_labels)
	train_features = [];	
	for j = 1:size(train_data,3)		    
		v = reshape(train_data(:,:,j),1,[]);
		train_features = [train_features, v'];
	end 

	train_labels = double(train_labels);

	a = randperm(10000);
	sets = {};
	set_labels = {};	
	max_train_set = train_features';
	max_train_labels = train_labels;

	for i = 1:10
		x = i * 1000;
		currSet = [];
		currLabel = [];
		for j = x - 1000+1: x
			currSet = [currSet; max_train_set(a(j),:)];
			currLabel = [currLabel; max_train_labels(a(j))];
		end
		set_labels{end + 1}  = currLabel;	
		sets{end + 1} = currSet;
	end
%	c = .5;

%	low = [0, 100]; 
%	high = [1, crossV(sets, set_labels, 1, 10)];
%	mid = [.01, crossV(sets, set_labels, .1, 10)];

%	mid_low = [];
%	mid_high  = [];

				
	%while abs(high(2) - low(2)) > .1
	%	mid
        %	mid_low  = [(mid(1) + low(1))/ 2,  crossV(sets, set_labels, (mid(1) + low(1))/ 2, 10)]
        %	mid_high  = [(mid(1) + high(1))/ 2,  crossV(sets, set_labels, (mid(1) + high(1))/ 2, 10)]
%
%		[min_error, boundary] = min([mid_low(2), mid(2), mid_high(2)]);
%		switch boundary
%			case 1
%				%low = [ (low[1] + midlow[1]) / 2, crossV(sets, set_labels, (low[1] + mid_low[1])/2, 10)];
%				high = mid;
%				mid = mid_low;
%			case 2
%				low = mid_low; 
%				high = mid_high; 
%	
%			case 3
%				low = mid;
%				mid = mid_high;
%				%high =  [ (high[1] + mid_high[1]) / 2, crossV(sets, set_labels, (mid[1] + mid_high[1])/2, 10)];
%		end	
%	end
 	
	best = [1, 100];
	curr = [1, 100];
	
	values = [];	
	for i = 0:10
		curr(1) = 10^-i;
		curr(2) = crossV(sets, set_labels, 10^-i, 10);				
		if best(2) > curr(2)
			best(1) = curr(1);
			best(2) = curr(2);
		end	
		values = [values; curr];
	end	
	best = values	
end


function [err] = crossV (sets, set_labels, c, k)

	err = 0;	

	sets;
	set_labels;	
	for i = 1:k
		hold_out = double(sets{i});
		hold_out_labels  = double(set_labels{i});
		validation = sets;

		validation(i) = [];
		validation = double(cell2mat(validation'));

		validation_labels = set_labels;
		validation_labels(i) = [];
		validation_labels = double(cell2mat(validation_labels'));

		s = sprintf('-c %.10f', c);
		[ model] = train( validation_labels, sparse(validation), s, '-q'); 
	
		[predicted_label_subset, acc, d] = predict(hold_out_labels, sparse(hold_out), model, '-q');
		err = err + 100 - acc(1);		
	end

	err = err/ k;

end 
