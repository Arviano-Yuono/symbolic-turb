//ccl: Function to print the genotype to logfile
void printGeno(
    wordList genotype,
	label  &i
	
){	
	if (genotype[i] == "N1"){
		i++; Info << "N1";
	}
	else if (genotype[i] == "N2"){
		i++; Info << "N2";
	}
	else if (genotype[i] == "N3"){
		i++; Info << "N3";
	} else if (genotype[i] == "I1"){
		i++;Info << "I1";
	} else if (genotype[i] == "I2") {
		i++; Info << "I2";
        } else if (genotype[i] == "*") {
		i++;
		Info << "(";
		printGeno(genotype, i);
		Info << " * ";
		printGeno(genotype, i);
		Info << ")";
	} else if (genotype[i] == "-") {
		i++;
	        Info << "(";
		printGeno(genotype, i);
		Info << " - ";
		printGeno(genotype, i);
		Info << ")";
	} else if (genotype[i] == "+") {
		i++;
		Info << "(";
		printGeno(genotype, i);
		Info << " + ";
		printGeno(genotype, i);
		Info << ")";
	}else if (genotype[i]=="exp"){	
		i++;
		Info << "exp(";
		printGeno(genotype, i);
		Info << ")";
	}else if (genotype[i]=="sqrt"){	
		i++;
		Info << "sqrt(";
		printGeno(genotype, i);
		Info << ")";
	}
	else if (genotype[i]=="sign"){	
		i++;
		Info << "sign(";
		printGeno(genotype, i);
		Info << ")";
	}
	else if (genotype[i]=="log"){	
		i++;
		Info << "log_abs(";
		printGeno(genotype, i);
		Info << ")";
	}
		else if (genotype[i]=="sin"){	
		i++;
		Info << "sin(";
		printGeno(genotype, i);
		Info << ")";
	}
		else if (genotype[i]=="cos"){	
		i++;
		Info << "cos(";
		printGeno(genotype, i);
		Info << ")";
	}
			else if (genotype[i]=="sinh"){	
		i++;
		Info << "sinh(";
		printGeno(genotype, i);
		Info << ")";
	}
			else if (genotype[i]=="cosh"){	
		i++;
		Info << "cosh(";
		printGeno(genotype, i);
		Info << ")";
	}
	else {
		i++;
		Info << genotype[i-1];
	}
}

//ccl: Function to evaluate the genotype
volScalarField evalGeno (
    wordList genotype,
	label  &i,
	volScalarField &I1,
	volScalarField &I2,
	volScalarField &N1,
	volScalarField &N2,
	volScalarField &N3
){
    volScalarField one("one", (I1+1e-10)/(I1+1e-10)); //ugly hack, cast to volScalarField :(
	if (genotype[i] == "N1"){
		i++;
		return N1;
	}
	else if (genotype[i] == "N2"){
		i++;
		return N2;
	}
	else if (genotype[i] == "N3"){
		i++;
		return N3;
	} else if (genotype[i] == "I1"){
		i++;
		return I1;
	} else if (genotype[i] == "I2"){
		i++; return I2;
        } else if (genotype[i] == "*"){
		i++;
		return evalGeno(genotype, i, I1, I2, N1, N2, N3) * evalGeno(genotype, i, I1, I2, N1, N2, N3);
	} else if (genotype[i] == "-"){
		i++;
                volScalarField tmp0("tmp0", evalGeno(genotype, i, I1, I2, N1, N2, N3));
                volScalarField tmp1("tmp1", evalGeno(genotype, i, I1, I2, N1, N2, N3));
                volScalarField tmp("tmp", tmp0 - tmp1);
                return tmp;
	} else if (genotype[i] == "+"){
		i++;
		return evalGeno(genotype, i, I1, I2, N1, N2, N3) + evalGeno(genotype, i, I1, I2, N1, N2, N3);
		
	}else if (genotype[i] == "sqrt"){
		i++;
		volScalarField tmp0("tmp0", evalGeno(genotype, i, I1, I2, N1, N2 ,N3));
		return Foam::sqrt(Foam::sign(tmp0)*tmp0);
		
	}
	else if (genotype[i]== "exp") {
		i++;
		return Foam::exp(evalGeno(genotype, i, I1, I2, N1, N2, N3));	
	}
	else if (genotype[i]== "sign") {
		i++;
		volScalarField tmp0("tmp0", evalGeno(genotype, i, I1, I2, N1, N2, N3));
		return Foam::sign(tmp0)*tmp0;	
	}
	else if (genotype[i]== "log") {
		i++;
		volScalarField tmp0("tmp0", evalGeno(genotype, i, I1, I2, N1, N2, N3));
		return Foam::log(Foam::neg0(tmp0));	
	}
	else if (genotype[i]== "sin") {
		i++;
		volScalarField tmp0("tmp0", evalGeno(genotype, i, I1, I2, N1, N2, N3));
		return Foam::sin(tmp0);	
	}
	else if (genotype[i]== "cos") {
		i++;
		volScalarField tmp0("tmp0", evalGeno(genotype, i, I1, I2, N1, N2, N3));
		return Foam::cos(tmp0);	
	}
		else if (genotype[i]== "sinh") {
		i++;
		volScalarField tmp0("tmp0", evalGeno(genotype, i, I1, I2, N1, N2, N3));
		return Foam::sinh(tmp0);	
	}
		else if (genotype[i]== "cosh") {
		i++;
		volScalarField tmp0("tmp0", evalGeno(genotype, i, I1, I2, N1, N2, N3));
		return Foam::cosh(tmp0);	
	}
	else {
		i++;
		char* pEnd;
		scalar numValue = strtod(genotype[i-1].c_str(),&pEnd);
		return numValue * one;
	}
	return one; //dummy return
}
