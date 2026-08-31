# Base model vs 1M-row QLoRA adapter

Sample size: 25 | Seed: 20260831

## Sample 1 (smileyllama)

**Instruction:** You love and excel at generating SMILES strings of drug-like molecules

**Input:** Output a SMILES string for a drug-like molecule with the following properties: no macrocycles, lacks bad SMARTS, <= 10 H-bond acceptors, lacks covalent warheads, <= 400 Molecular weight, <= 4 H-bond donors:

**Reference:**
```text
COc1ccccc1CN1CCN(C(COc2cccc(c2)C)=O)CC1
```

**Base:**
```text
Here is a SMILES string for a drug-like molecule that meets the specified properties:

CC(=O)Nc1ccc(cc1)C(=O)N

This SMILES string represents a molecule with the following properties:

*   No macrocycles
*   Lacks bad SMARTS
*   <= 10 H-bond acceptors
*   Lacks covalent warheads
*   <= 400 Molecular weight
*   <= 4 H-bond donors

This molecule is a drug-like compound with a molecular weight of 276.32 g/mol, which is within the specified limit. It also meets the other criteria, including the absence of macrocycles, bad SMARTS, and covalent warheads. The molecule has 4 H-bond donors and 10 H-bond acceptors, which are within the specified limits. 

Please note that the SMILES string provided is a simple example and may not represent a real-world drug molecule. The properties of the molecule can be adjusted to meet specific requirements. 

If you need any further assistance or have any questions, feel free to ask! 

### Note:
The SMILES string provided is a simple example and may not represent a real-world drug molecule. The properties of the molecule
```

**QLoRA:**
```text
c1c(ccc(c1)C(=O)Nc1ccccc1)C(=O)Nc1ccccc1
```

## Sample 2 (smileyllama)

**Instruction:** You love and excel at generating SMILES strings of drug-like molecules

**Input:** Output a SMILES string for a drug-like molecule with the following properties: <= 500 Molecular weight, a substructure of Oc1ccc(Nc2ccccn2)cc1, no macrocycles, <= 0.4 Fraction sp3, <= 140 TPSA, <= 10 H-bond acceptors, <= 3 H-bond donors, lacks covalent warheads:

**Reference:**
```text
c1(Oc2c(N3CCC(C3)c3onc(C)n3)cccn2)ccc(cc1)Nc1ccccn1
```

**Base:**
```text
Here is a SMILES string that meets the specified properties:

CC(=O)Nc1ccc(Nc2ccccn2)cc1

This SMILES string represents a molecule with the following characteristics:

*   Molecular weight: 246.3
*   Substructure: Oc1ccc(Nc2ccccn2)cc1
*   No macrocycles
*   Fraction sp3: 0.38
*   TPSA: 114.3
*   H-bond acceptors: 4
*   H-bond donors: 1
*   Lacks covalent warheads

This molecule meets all the specified criteria, including a molecular weight of 246.3, which is less than or equal to 500. It also contains the specified substructure, Oc1ccc(Nc2ccccn2)cc1, and lacks macrocycles. Additionally, it has a fraction of sp3 hybridized atoms of 0.38, which is less than or equal to 0.4. The TPSA is 114.3, which is less than or equal to 140. The molecule has 4 H-bond acceptors and 1 H-bond donor, both of which are less
```

**QLoRA:**
```text
c1c(ccc(c1)O)Nc1ncc(cc1)Nc1ccc(cc1)O
```

## Sample 3 (synllama)

**Instruction:** You are an expert synthetic organic chemist. Your task is to design a synthesis pathway for a given target molecule using common and reliable reaction templates and building blocks. Follow these instructions:

1. **Input the SMILES String:** Read in the SMILES string of the target molecule and identify common reaction templates that can be applied.

2. **Decompose the Target Molecule:** Use the identified reaction templates to decompose the target molecule into different intermediates.

3. **Check for Building Blocks:** For each intermediate:
   - Identify if it is a building block. If it is, wrap it in <bb> and </bb> tags and save it for later use.
   - If it is not a building block, apply additional reaction templates to further decompose it into building blocks.

4. **Document Reactions:** For each reaction documented in the output, wrap the reaction template in <rxn> and </rxn> tags.

5. **Repeat the Process:** Continue this process until all intermediates are decomposed into building blocks, and document each step clearly in a structured JSON format.

**Input:** Provide a synthetic pathway for this SMILES string: Cc1cnn(-c2cc3c(=O)n(-c4ccn(C)c(=O)c4)c(C(Cc4ccc5ccccc5n4)NC(=O)NCCC4=C(C5(C(=O)ON6C(=O)c7ccccc7C6=O)CCCN(C(=O)OC(C)(C)C)C5)Nc5cc(Br)c(Cl)cc54)nc3cc2F)c1

**Reference:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[NH1;$(N-c1ccccc1):1](-[NH2])-[c:5]:[cH1:4].[C;$(C([#6])[#6]):2](=[OD1])-[CH2;$(C([#6])[#6]);!$(C(C=O)C=O):3]>>[C:5]1-[N:1]-[C:2]=[C:3]-[C:4]:1</rxn>", "reactants": ["", "Cc1cnn(-c2cc3c(=O)n(-c4ccn(C)c(=O)c4)c(C(Cc4ccc5ccccc5n4)NC(=O)NCCCC(=O)C4(C(=O)ON5C(=O)c6ccccc6C5=O)CCCN(C(=O)OC(C)(C)C)C4)nc3cc2F)c1", "NNc1ccc(Cl)c(Br)c1"], "product": "Cc1cnn(-c2cc3c(=O)n(-c4ccn(C)c(=O)c4)c(C(Cc4ccc5ccccc5n4)NC(=O)NCCC4=C(C5(C(=O)ON6C(=O)c7ccccc7C6=O)CCCN(C(=O)OC(C)(C)C)C5)Nc5cc(Br)c(Cl)cc54)nc3cc2F)c1"}, {"reaction_number": 2, "reaction_template": "<rxn>[C$([C](O)([CX4])([CX4])([CX4])),C$([CH](O)([CX4])([CX4])),C$([CH2](O)([CX4])):4]-[O:3]-[C$(C(=O)([CX4])),C$([CH](=O)):2]=[O:5].[C$([CH](C)([CX4])([CX4])),C$([CH2](C)([CX4])),C$([CH3](C)):7]-[C$(C(=O)([CX4])),C$([CH](=O)):8]=[O:9]>>[C:7](-[C:2]=[O:5])-[C:8]=[O:9]</rxn>", "reactants": ["", "Cc1cnn(-c2cc3c(cc2F)N=C(C(Cc2ccc4ccccc4n2)NC(=O)NCCCC(=O)OC(C)(C)C)N(c2ccn(C)c(=O)c2)C3=O)c1", "CC(C)(C)OC(=O)N1CCCC(C(=O)ON2C(=O)c3ccccc3C2=O)C1"], "product": "Cc1cnn(-c2cc3c(=O)n(-c4ccn(C)c(=O)c4)c(C(Cc4ccc5ccccc5n4)NC(=O)NCCCC(=O)C4(C(=O)ON5C(=O)c6ccccc6C5=O)CCCN(C(=O)OC(C)(C)C)C4)nc3cc2F)c1"}, {"reaction_number": 3, "reaction_template": "<rxn>[c:1](-[C;$(C-c1ccccc1):2](=[OD1:3])-[OH1]):[c:4](-[NH2:5]).[N;!H0;!$(N-N);!$(N-C=N);!$(N(-C=O)-C=O):6]-[C;H1,$(C-[#6]):7]=[OD1]>>[c:4]2:[c:1]-[C:2](=[O:3])-[N:6]-[C:7]=[N:5]-2</rxn>", "reactants": ["", "Cn1ccc(NC(=O)C(Cc2ccc3ccccc3n2)NC(=O)NCCCC(=O)OC(C)(C)C)cc1=O", "Cc1cnn(-c2cc(C(=O)O)c(N)cc2F)c1"], "product": "Cc1cnn(-c2cc3c(cc2F)N=C(C(Cc2ccc4ccccc4n2)NC(=O)NCCCC(=O)OC(C)(C)C)N(c2ccn(C)c(=O)c2)C3=O)c1"}, {"reaction_number": 4, "reaction_template": "<rxn>[Cl,OH,O-:3][C$(C(=O)([CX4,c])),C$([CH](=O)):2]=[O:4].[N$([NH2,NH3+1]([CX4,c])),N$([NH]([CX4,c])([CX4,c])):6]>>[N+0:6]-[C:2]=[O:4]</rxn>", "reactants": ["", "CC(C)(C)OC(=O)CCCNC(=O)NC(Cc1ccc2ccccc2n1)C(=O)O", "Cn1ccc(N)cc1=O"], "product": "Cn1ccc(NC(=O)C(Cc2ccc3ccccc3n2)NC(=O)NCCCC(=O)OC(C)(C)C)cc1=O"}, {"reaction_number": 5, "reaction_template": "<rxn>[N;$(N-[#6]):3]=[C;$(C=O):1].[N;$(N[#6]);!$(N=*);!$([N-]);!$(N#*);!$([ND3]);!$([ND4]);!$(N[O,N]);!$(N[C,S]=[S,O,N]):2]>>[N:3]-[C:1]-[N+0:2]</rxn>", "reactants": ["", "NC(Cc1ccc2ccccc2n1)C(=O)O", "CC(C)(C)OC(=O)CCCN=C=O"], "product": "CC(C)(C)OC(=O)CCCNC(=O)NC(Cc1ccc2ccccc2n1)C(=O)O"}], "building_blocks": ["<bb>NNc1ccc(Cl)c(Br)c1</bb>", "<bb>CC(C)(C)OC(=O)N1CCCC(C(=O)ON2C(=O)c3ccccc3C2=O)C1</bb>", "<bb>Cc1cnn(-c2cc(C(=O)O)c(N)cc2F)c1</bb>", "<bb>Cn1ccc(N)cc1=O</bb>", "<bb>CC(C)(C)OC(=O)CCCN=C=O</bb>", "<bb>NC(Cc1ccc2ccccc2n1)C(=O)O</bb>"]}
```

**Base:**
```text
```json
{
  "synthetic_pathway": [
    {
      "target": "Cc1cnn(-c2cc3c(=O)n(-c4ccn(C)c(=O)c4)c(C(Cc4ccc5ccccc5n4)NC(=O)NCCC4=C(C5(C(=O)ON6C(=O)c7ccccc7C6=O)CCCN(C(=O)OC(C)(C)C)C5)Nc5cc(Br)c(Cl)cc54)nc3cc2F)c1",
      "intermediates": [
        {
          "name": "Intermediate 1",
          "smiles": "Cc1cnn(-c2cc3c(=O)n(-c4ccn(C)c(=O)c4)c(C(Cc4ccc5ccccc5n4)NC(=O)NCCC4=C(C5(C(=O)ON6C(=O)c7ccccc7C6=O)CCCN(C(=O)OC(C)(C)C)C5)Nc5cc(Br)c(Cl)cc54)nc3cc2F)c1",
          "building_block": false,
          "
```

**QLoRA:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[NH1;$(N-c1ccccc1):1](-[NH2])-[c:5]:[cH1:4].[C;$(C([#6])[#6]):2](=[OD1])-[CH2;$(C([#6])[#6]);!$(C(C=O)C=O):3]>>[C:5]1-[N:1]-[C:2]=[C:3]-[C:4]:1</rxn>", "reactants": ["", "Cc1cnn(-c2cc3c(=O)n(-c4ccn(C)c(=O)c4)c(C(Cc4ccc5ccccc5n4)NC(=O)NCCCC(=O)C4(C(=O)ON5C(=O)c6ccccc6C5=O)CCCN(C(=O)OC(C)(C)C)C4)nc3cc2F)c1", "NNc1ccc(Cl)c(Br)c1"], "product": "Cc1cnn(-c2cc3c(=O)n(-c4ccn(C
```

## Sample 4 (smileyllama)

**Instruction:** You love and excel at generating SMILES strings of drug-like molecules

**Input:** Output a SMILES string for a drug-like molecule with the following properties: <= 140 TPSA, <= 10 H-bond acceptors, lacks covalent warheads, lacks bad SMARTS, no macrocycles, <= 600 Molecular weight, <= 10 Rotatable bonds, <= 3 H-bond donors, A formula of C25H26ClN5O3S:

**Reference:**
```text
c1(CNC(c2c(=O)c3sc(CN(CC(O)c4nnccc4)C)c(C)c3n(c2)C)=O)ccc(cc1)Cl
```

**Base:**
```text
CC(=O)N1CCC2=C(C3=C(C(=C(C4=C(C5=C(C(=C6C(=C(C(=C7C(=C(C(=C8C(=C(C9=C(C(=C(C10=C(C(=C(C11=C(C(=C(C12=C(C(=C(C13=C(C(=C(C14=C(C(=C(C15=C(C(=C(C16=C(C(=C(C17=C(C(=C(C18=C(C(=C(C19=C(C(=C(C20=C(C(=C(C21=C(C(=C(C22=C(C(=C(C23=C(C(=C(C24=C(C(=C(C25=C(C(=C(C26=C(C(=C(C27=C(C(=C(C28=C(C(=C(C29=C(C(=C(C30=C(C(=C(C31=C(C(=C(C32=C(C(=C(C33=C(C(=C(C34=C(C(=C(C35=C(C(=C(C36=C(C(=C(C37=C(C(=C(C38=C(C(=C(C39=C(C(=C(C40=C(C(=C(C41=C(C(=C(C42=C(C(=C(C43=C(C(=C(C44=C(C
```

**QLoRA:**
```text
c1c(ccc(c1)Cl)C1CCN(C(=O)CSc2nnc(n2-c2ccc(cc2)OC)C)CC1C(N)=O
```

## Sample 5 (synllama)

**Instruction:** You are an expert synthetic organic chemist. Your task is to design a synthesis pathway for a given target molecule using common and reliable reaction templates and building blocks. Follow these instructions:

1. **Input the SMILES String:** Read in the SMILES string of the target molecule and identify common reaction templates that can be applied.

2. **Decompose the Target Molecule:** Use the identified reaction templates to decompose the target molecule into different intermediates.

3. **Check for Building Blocks:** For each intermediate:
   - Identify if it is a building block. If it is, wrap it in <bb> and </bb> tags and save it for later use.
   - If it is not a building block, apply additional reaction templates to further decompose it into building blocks.

4. **Document Reactions:** For each reaction documented in the output, wrap the reaction template in <rxn> and </rxn> tags.

5. **Repeat the Process:** Continue this process until all intermediates are decomposed into building blocks, and document each step clearly in a structured JSON format.

**Input:** Provide a synthetic pathway for this SMILES string: CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4(C(C)(OCC(=O)C(C#N)=C5N(C)c6ccccc6N5C)c5ccc(-c6cc(NC7CC8CCC(C7C)N8C(=O)OC(C)(C)C)n(C)n6)cn5)CCCC4)CC3)cc2[nH]1

**Reference:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[#6;$([#6]~[#6]);!$([#6]=O):2][#8;H1:3].[Cl,Br,I][#6;H2;$([#6]~[#6]):4]>>[CH2:4][O:3][#6:2]</rxn>", "reactants": ["", "CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4(C(C)(O)c5ccc(-c6cc(NC7CC8CCC(C7C)N8C(=O)OC(C)(C)C)n(C)n6)cn5)CCCC4)CC3)cc2[nH]1", "CN1C(=C(C#N)C(=O)CCl)N(C)c2ccccc21"], "product": "CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4(C(C)(OCC(=O)C(C#N)=C5N(C)c6ccccc6N5C)c5ccc(-c6cc(NC7CC8CCC(C7C)N8C(=O)OC(C)(C)C)n(C)n6)cn5)CCCC4)CC3)cc2[nH]1"}, {"reaction_number": 2, "reaction_template": "<rxn>[C$(C(=O)([CX4,c])([CX4,c])),C$([CH](=O)([CX4,c])):1]=[O:2].[N$([NH2,NH3+1]([CX4,c])),N$([NH]([CX4,c])([CX4,c])):3]>>[N+0:3][C:1]</rxn>", "reactants": ["", "CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4(C(C)(O)c5ccc(-c6cc(N)n(C)n6)cn5)CCCC4)CC3)cc2[nH]1", "CC1C(=O)CC2CCC1N2C(=O)OC(C)(C)C"], "product": "CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4(C(C)(O)c5ccc(-c6cc(NC7CC8CCC(C7C)N8C(=O)OC(C)(C)C)n(C)n6)cn5)CCCC4)CC3)cc2[nH]1"}, {"reaction_number": 3, "reaction_template": "<rxn>[#6:1][C;H1,$([C]([#6])[#6]):2]=[OD1:3].[Cl,Br,I][#6;$([#6]~[#6]);!$([#6]([Cl,Br,I])[Cl,Br,I]);!$([#6]=O):4]>>[C:1][#6:2]([OH1:3])[#6:4]</rxn>", "reactants": ["", "CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4(C(C)=O)CCCC4)CC3)cc2[nH]1", "Cn1nc(-c2ccc(Cl)nc2)cc1N"], "product": "CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4(C(C)(O)c5ccc(-c6cc(N)n(C)n6)cn5)CCCC4)CC3)cc2[nH]1"}, {"reaction_number": 4, "reaction_template": "<rxn>[C$([C](O)([CX4])([CX4])([CX4])),C$([CH](O)([CX4])([CX4])),C$([CH2](O)([CX4])):4]-[O:3]-[C$(C(=O)([CX4])),C$([CH](=O)):2]=[O:5].[C$([CH](C)([CX4])([CX4])),C$([CH2](C)([CX4])),C$([CH3](C)):7]-[C$(C(=O)([CX4])),C$([CH](=O)):8]=[O:9]>>[C:7](-[C:2]=[O:5])-[C:8]=[O:9]</rxn>", "reactants": ["", "CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4CCCC4)CC3)cc2[nH]1", "O=C1CC(CBr)O1"], "product": "CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4(C(C)=O)CCCC4)CC3)cc2[nH]1"}, {"reaction_number": 5, "reaction_template": "<rxn>[NH2,NH3+1:8]-[c:5]1[cH:4][c:3][c:2][c:1][c:6]1.[Br:18][C$([CH2](C)(Br)),C$([CH](C)(C)(Br)):17]-[C:15](=[O:16])-[c:10]1[c:11][c:12][c:13][c:14][c:9]1>>[c:13]1[c:12][c:11][c:10]([c:9][c:14]1)-[c:15]1[c:17][c:4]2[c:3][c:2][c:1][c:6][c:5]2[nH+0:8]1</rxn>", "reactants": ["", "Nc1cccc(S(=O)(=O)N2CCN(C(=O)C3CCCC3)CC2)c1", "CCCCCCCCc1ccccc1C(=O)CBr"], "product": "CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4CCCC4)CC3)cc2[nH]1"}], "building_blocks": ["<bb>CN1C(=C(C#N)C(=O)CCl)N(C)c2ccccc21</bb>", "<bb>CC1C(=O)CC2CCC1N2C(=O)OC(C)(C)C</bb>", "<bb>Cn1nc(-c2ccc(Cl)nc2)cc1N</bb>", "<bb>O=C1CC(CBr)O1</bb>", "<bb>CCCCCCCCc1ccccc1C(=O)CBr</bb>", "<bb>Nc1cccc(S(=O)(=O)N2CCN(C(=O)C3CCCC3)CC2)c1</bb>"]}
```

**Base:**
```text
```json
{
  "synthesis_pathway": [
    {
      "target": "CCCCCCCCC1CC2CCC(S(=O)(=O)N3CCN(C(=O)C4(C(C)(OCC(=O)C(C#N)=C5N(C)c6ccccc6N5C)c5ccc(-c6cc(NC7CC8CCC(C7C)N8C(=O)OC(C)(C)C)n(C)n6)cn5)CCCC4)CC3)CC2[nH]1",
      "intermediates": [
        {
          "name": "CCCCCCCCC1CC2CCC(S(=O)(=O)N3CCN(C(=O)C4(C(C)(OCC(=O)C(C#N)=C5N(C)c6ccccc6N5C)c5ccc(-c6cc(NC7CC8CCC(C7C)N8C(=O)OC(C)(C)C)n(C)n6)cn5)CCCC4)CC3)CC2[nH]1",
          "building_block": false,
          "reaction_templates": [
            {
              "template": "C(=O)C
```

**QLoRA:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[#6;$([#6]~[#6]);!$([#6]=O):2][#8;H1:3].[Cl,Br,I][#6;H2;$([#6]~[#6]):4]>>[CH2:4][O:3][#6:2]</rxn>", "reactants": ["", "CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4(C(C)(O)c5ccc(-c6cc(NC7CC8CCC(C7C)N8C(=O)OC(C)(C)C)n(C)n6)cn5)CCCC4)CC3)cc2[nH]1", "CN1C(=C(C#N)C(=O)CCl)N(C)c2ccccc21"], "product": "CCCCCCCCc1ccccc1-c1cc2ccc(S(=O)(=O)N3CCN(C(=O)C4(C(C)(OCC(=O)C(C#N)=C5N(C)c6cc
```

## Sample 6 (synllama)

**Instruction:** You are an expert synthetic organic chemist. Your task is to design a synthesis pathway for a given target molecule using common and reliable reaction templates and building blocks. Follow these instructions:

1. **Input the SMILES String:** Read in the SMILES string of the target molecule and identify common reaction templates that can be applied.

2. **Decompose the Target Molecule:** Use the identified reaction templates to decompose the target molecule into different intermediates.

3. **Check for Building Blocks:** For each intermediate:
   - Identify if it is a building block. If it is, wrap it in <bb> and </bb> tags and save it for later use.
   - If it is not a building block, apply additional reaction templates to further decompose it into building blocks.

4. **Document Reactions:** For each reaction documented in the output, wrap the reaction template in <rxn> and </rxn> tags.

5. **Repeat the Process:** Continue this process until all intermediates are decomposed into building blocks, and document each step clearly in a structured JSON format.

**Input:** Provide a synthetic pathway for this SMILES string: CCn1c(Cn2cc(-c3ccc4c(c3)nc(C3(N5N=NN=C5c5ccc(F)c(C=O)c5O)CCCC5C3C5(F)F)n4C)nc2C)nc2ccc([N+](=O)[O-])cc21

**Reference:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[CH0;$(C-[#6]):1]#[NH0:2].[C;A;!$(C=O):3]-[*;#17,#35,#53]>>[C:1]1=[N:2]-N=N-N-1(-[C:3])</rxn>", "reactants": ["", "CCn1c(Cn2cc(-c3ccc4c(c3)nc(C3(Cl)CCCC5C3C5(F)F)n4C)nc2C)nc2ccc([N+](=O)[O-])cc21", "N#Cc1ccc(F)c(C=O)c1O"], "product": "CCn1c(Cn2cc(-c3ccc4c(c3)nc(C3(N5N=NN=C5c5ccc(F)c(C=O)c5O)CCCC5C3C5(F)F)n4C)nc2C)nc2ccc([N+](=O)[O-])cc21"}, {"reaction_number": 2, "reaction_template": "<rxn>[c;r6:1](-[NH1;$(N-[#6]):2]):[c;r6:3](-[NH2:4]).[#6:6]-[C;R0:5](=[OD1])-[#8;H1,$(O-[CH3])]>>[c:3]2:[c:1]:[n:2]:[c:5](-[#6:6]):[n:4]2</rxn>", "reactants": ["", "COC(=O)Cn1cc(-c2ccc3c(c2)nc(C2(Cl)CCCC4C2C4(F)F)n3C)nc1C", "CCNc1cc([N+](=O)[O-])ccc1N"], "product": "CCn1c(Cn2cc(-c3ccc4c(c3)nc(C3(Cl)CCCC5C3C5(F)F)n4C)nc2C)nc2ccc([N+](=O)[O-])cc21"}, {"reaction_number": 3, "reaction_template": "<rxn>[Br,I:1][C$(C([Br,I])([CX4])([CX4])([CX4])),C$([CH]([Br,I])([CX4])([CX4])),C$([CH2]([Br,I])([CX4])),C$([CH3]([Br,I])),C$([C]([Br,I])(=C)([CX4])),C$([CH]([Br,I])(=C)),C$(C([Br,I])(#C)),c$(c([Br,I])):2].[Br,I:3][C$(C([Br,I])([CX4])([CX4])([CX4])),C$([CH]([Br,I])([CX4])([CX4])),C$([CH2]([Br,I])([CX4])),C$([CH3]([Br,I])),C$([C]([Br,I])(=C)([CX4])),C$([CH]([Br,I])(=C)),C$(C([Br,I])(#C)),c$(c([Br,I])):4]>>[C,c:2][C,c:4]</rxn>", "reactants": ["", "Cn1c(C2(Cl)CCCC3C2C3(F)F)nc2cc(I)ccc21", "COC(=O)Cn1cc(Br)nc1C"], "product": "COC(=O)Cn1cc(-c2ccc3c(c2)nc(C2(Cl)CCCC4C2C4(F)F)n3C)nc1C"}, {"reaction_number": 4, "reaction_template": "<rxn>[c;r6:1](-[NH1;$(N-[#6]):2]):[c;r6:3](-[NH2:4]).[#6:6]-[C;R0:5](=[OD1])-[#8;H1,$(O-[CH3])]>>[c:3]2:[c:1]:[n:2]:[c:5](-[#6:6]):[n:4]2</rxn>", "reactants": ["", "O=C(O)C1(Cl)CCCC2C1C2(F)F", "CNc1ccc(I)cc1N"], "product": "Cn1c(C2(Cl)CCCC3C2C3(F)F)nc2cc(I)ccc21"}, {"reaction_number": 5, "reaction_template": "<rxn>[OH+0,O-:5]-[C:3](=[O:4])-[C$([CH]([CX4])),C$([CH2]):2]>>[OH+0,O-:5]-[C:3](=[O:4])-[C:2]([Cl:6])</rxn>", "reactants": ["", "", "O=C(O)C1CCCC2C1C2(F)F"], "product": "O=C(O)C1(Cl)CCCC2C1C2(F)F"}], "building_blocks": ["<bb>N#Cc1ccc(F)c(C=O)c1O</bb>", "<bb>CCNc1cc([N+](=O)[O-])ccc1N</bb>", "<bb>COC(=O)Cn1cc(Br)nc1C</bb>", "<bb>CNc1ccc(I)cc1N</bb>", "<bb>O=C(O)C1CCCC2C1C2(F)F</bb>"]}
```

**Base:**
```text
```json
{
  "synthesis_pathway": [
    {
      "target": "CCn1c(Cn2cc(-c3ccc4c(c3)nc(C3(N5N=NN=C5c5ccc(F)c(C=O)c5O)CCCC5C3C5(F)F)n4C)nc2C)nc2ccc([N+](=O)[O-])cc21",
      "intermediates": [
        {
          "name": "Intermediate 1",
          "smiles": "CCn1c(Cn2cc(-c3ccc4c(c3)nc(C3(N5N=NN=C5c5ccc(F)c(C=O)c5O)CCCC5C3C5(F)F)n4C)nc2C)nc2ccc([N+](=O)[O-])cc21",
          "building_block": false,
          "reaction": {
            "template": "Ullmann Coupling",
            "reactants": ["CCn1c(Cn2cc(-c3ccc4c(c3)nc(C3(N5N=NN=C5c5ccc(F)c(C=O)c5O)CCCC
```

**QLoRA:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[CH0;$(C-[#6]):1]#[NH0:2].[C;A;!$(C=O):3]-[*;#17,#35,#53]>>[C:1]1=[N:2]-N=N-N-1(-[C:3])</rxn>", "reactants": ["", "CCn1c(Cn2cc(-c3ccc4c(c3)nc(C3(Cl)CCCC5C3C5(F)F)n4C)nc2C)nc2ccc([N+](=O)[O-])cc21", "N#Cc1ccc(F)c(C=O)c1O"], "product": "CCn1c(Cn2cc(-c3ccc4c(c3)nc(C3(N5N=NN=C5c5ccc(F)c(C=O)c5O)CCCC5C3C5(F)F)n4C)nc2C)nc2ccc([N+](=O)[O-])cc21"}, {"reaction_number": 2, "reaction_template": "<rxn>[c;r6:1
```

## Sample 7 (synllama)

**Instruction:** You are an expert synthetic organic chemist. Your task is to design a synthesis pathway for a given target molecule using common and reliable reaction templates and building blocks. Follow these instructions:

1. **Input the SMILES String:** Read in the SMILES string of the target molecule and identify common reaction templates that can be applied.

2. **Decompose the Target Molecule:** Use the identified reaction templates to decompose the target molecule into different intermediates.

3. **Check for Building Blocks:** For each intermediate:
   - Identify if it is a building block. If it is, wrap it in <bb> and </bb> tags and save it for later use.
   - If it is not a building block, apply additional reaction templates to further decompose it into building blocks.

4. **Document Reactions:** For each reaction documented in the output, wrap the reaction template in <rxn> and </rxn> tags.

5. **Repeat the Process:** Continue this process until all intermediates are decomposed into building blocks, and document each step clearly in a structured JSON format.

**Input:** Provide a synthetic pathway for this SMILES string: COc1ccc(CNC(=S)N(CC(COCc2cn3c(C)cccc3n2)Cc2ccccc2C(F)(F)F)c2ccc(C(C)(F)F)cc2[N+](=O)[O-])cc1

**Reference:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[N;$(N-[#6]):3]=[C;$(C=S):1].[N;$(N[#6]);!$(N=*);!$([N-]);!$(N#*);!$([ND3]);!$([ND4]);!$(N[O,N]);!$(N[C,S]=[S,O,N]):2]>>[N:3]-[C:1]-[N+0:2]</rxn>", "reactants": ["", "Cc1cccc2nc(COCC(CNc3ccc(C(C)(F)F)cc3[N+](=O)[O-])Cc3ccccc3C(F)(F)F)cn12", "COc1ccc(CN=C=S)cc1"], "product": "COc1ccc(CNC(=S)N(CC(COCc2cn3c(C)cccc3n2)Cc2ccccc2C(F)(F)F)c2ccc(C(C)(F)F)cc2[N+](=O)[O-])cc1"}, {"reaction_number": 2, "reaction_template": "<rxn>[#6;$([#6]~[#6]);!$([#6]=O):2][#8;H1:3].[Cl,Br,I][#6;H2;$([#6]~[#6]):4]>>[CH2:4][O:3][#6:2]</rxn>", "reactants": ["", "CC(F)(F)c1ccc(NCC(CO)Cc2ccccc2C(F)(F)F)c([N+](=O)[O-])c1", "Cc1cccc2nc(CBr)cn12"], "product": "Cc1cccc2nc(COCC(CNc3ccc(C(C)(F)F)cc3[N+](=O)[O-])Cc3ccccc3C(F)(F)F)cn12"}, {"reaction_number": 3, "reaction_template": "<rxn>[c;$(c1c(N(~O)~O)cccc1):1][Cl,F].[N;$(NC);!$(N=*);!$([N-]);!$(N#*);!$([ND3]);!$([ND4]);!$(N[c,O]);!$(N[C,S]=[S,O,N]):2]>>[c:1][N:2]</rxn>", "reactants": ["", "NCC(CO)Cc1ccccc1C(F)(F)F", "CC(F)(F)c1ccc(Cl)c([N+](=O)[O-])c1"], "product": "CC(F)(F)c1ccc(NCC(CO)Cc2ccccc2C(F)(F)F)c([N+](=O)[O-])c1"}], "building_blocks": ["<bb>COc1ccc(CN=C=S)cc1</bb>", "<bb>Cc1cccc2nc(CBr)cn12</bb>", "<bb>CC(F)(F)c1ccc(Cl)c([N+](=O)[O-])c1</bb>", "<bb>NCC(CO)Cc1ccccc1C(F)(F)F</bb>"]}
```

**Base:**
```text
```json
{
  "synthetic_pathway": [
    {
      "target": "COc1ccc(CNC(=S)N(CC(COCc2cn3c(C)cccc3n2)Cc2ccccc2C(F)(F)F)c2ccc(C(C)(F)F)cc2[N+](=O)[O-])cc1",
      "intermediates": [
        {
          "name": "intermediate_1",
          "smiles": "COCc2cn3c(C)cccc3n2",
          "building_block": false,
          "reactions": [
            {
              "template": "Suzuki Coupling",
              "reactants": ["COCc2cn3c(C)cccc3n2", "CC(COCc2cn3c(C)cccc3n2)Cc2ccccc2C(F)(F)F"],
              "products": ["CC(COCc2cn3c(C)cccc3n2)Cc2ccccc2C(F)(F)F"]
            }
          ]
        },
        {
          "name": "intermediate_2",
          "smiles": "CC(COCc2cn3c
```

**QLoRA:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[N;$(N-[#6]):3]=[C;$(C=S):1].[N;$(N[#6]);!$(N=*);!$([N-]);!$(N#*);!$([ND3]);!$([ND4]);!$(N[O,N]);!$(N[C,S]=[S,O,N]):2]>>[N:3]-[C:1]-[N+0:2]</rxn>", "reactants": ["", "Cc1cccc2nc(COCC(CNc3ccc(C(C)(F)F)cc3[N+](=O)[O-])Cc3ccccc3C(F)(F)F)cn12", "COc1ccc(CN=C=S)cc1"], "product": "COc1ccc(CNC(=S)N(CC(COCc2cn3c(C)cccc3n2)Cc2ccccc2C(F)(F)F)c2ccc(C(C)(F)F)cc2[N+](=O)[O-])cc1"}, {"reaction_number": 2, "reaction_template": "<rx
```

## Sample 8 (linkllama)

**Instruction:** You are an expert medicinal chemist specializing in linker design. Your task is to design a linker to connect given fragments and deduce whether the final molecule is chemically reasonable. The output should be in JSON format.

**Input:** Fragment 1 (SMILES: CCCCCC(=O)N(CCCC)[*:2]) and Fragment 2 (SMILES: NCC1OC(OC2C(N)CC(N)C([*:1])C2O)C(N)C(O)C1O). The distance between the attachment points is 6.11 Angstroms, and the angle between them is 109.17 degrees. Given the above information about the fragments and attachment points, design a linker to connect them. The final molecule should be unreasonable. And it should have the following properties: > 5 H-bond donors, <= 700 Molecular weight.

**Reference:**
```text
{"linker": "NC1C(O)C(C[*:2])OC(O[*:1])C1O", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: Aliphatic long chain."}
```

**Base:**
```text
```json
{
  "linker": {
    "smiles": "C1CCC2C(=O)N(CCCC)C3=CC=CC=C3C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O)N(C)C(=O
```

**QLoRA:**
```text
{"linker": "NC1CC(N)C(O[*:1])C(O)C1[*:2]", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: Aliphatic long chain."}
```

## Sample 9 (linkllama)

**Instruction:** You are an expert medicinal chemist specializing in linker design. Your task is to design a linker to connect given fragments and deduce whether the final molecule is chemically reasonable. The output should be in JSON format.

**Input:** Fragment 1 (SMILES: CC(C)[C@H]1/C=C/C(=O)NCC/C=C/[C@H]([*:2])C(=O)N1) and Fragment 2 (SMILES: CCCCCCCCCC(=O)N[*:1]). The distance between the attachment points is 4.90 Angstroms, and the angle between them is 112.47 degrees. Given the above information about the fragments and attachment points, design a linker with >= 4 heavy atoms to connect them. The final molecule should be unreasonable. And it should have the following properties: <= 5 H-bond acceptors, <= 200 TPSA.

**Reference:**
```text
{"linker": "O=C(N[*:2])[C@H](Cc1ccccc1Cl)[*:1]", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: Aliphatic long chain."}
```

**Base:**
```text
```json
{
  "linker": {
    "smiles": "C1=CC=C(C(=O)NCC/C=C/C(=O)N[*:1])C(=O)N[*:2]C2=CC=CC=C2",
    "heavy_atoms": 7,
    "h_bond_acceptors": 4,
    "tpsa": 120.5,
    "unreasonable": true
  }
}
```
### Explanation:
The task requires designing a linker to connect two given fragments. The linker should have at least 4 heavy atoms, be chemically unreasonable, have <= 5 H-bond acceptors, and <= 200 TPSA. 

The designed linker connects the two fragments with a chain of 7 heavy atoms, which meets the requirement of having at least 4 heavy atoms. The linker also has 4 H-bond acceptors, which meets the requirement of having <= 5 H-bond acceptors. The TPSA of the linker is 120.5, which is within the allowed limit of <= 200 TPSA.

However, the final molecule is chemically unreasonable because the linker is too long and the angle between the attachment points is too small,
```

**QLoRA:**
```text
{"linker": "O=C(N[*:2])[C@H](Cc1ccc(O)cc1)[*:1]", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: Aliphatic long chain."}
```

## Sample 10 (smileyllama)

**Instruction:** You love and excel at generating SMILES strings of drug-like molecules

**Input:** Output a SMILES string for a drug-like molecule with the following properties: <= 15 H-bond acceptors, <= 90 TPSA, > 0.4 Fraction sp3, has bad SMARTS, <= 6 LogP, <= 500 Molecular weight, lacks covalent warheads:

**Reference:**
```text
c12c3c(c4c(n(CCC(C)C)c5c4cc(C4CCCCO4)cc5)c1CCc1nn(cc-21)C)CNC3=O
```

**Base:**
```text
Here is a SMILES string that meets the specified criteria:

`CC(=O)Nc1ccc(cc1)C(=O)N`

This SMILES string represents a molecule with the following properties:

*   <= 15 H-bond acceptors: 4
*   <= 90 TPSA: 73.3
*   > 0.4 Fraction sp3: 0.73
*   has bad SMARTS: Yes
*   <= 6 LogP: 2.1
*   <= 500 Molecular weight: 221.26
*   lacks covalent warheads: Yes

This molecule meets all the specified criteria. However, please note that the SMILES string is just an example and may not represent a real-world drug-like molecule. The properties mentioned above are based on the given SMILES string and may not reflect the actual properties of a real-world molecule. 

If you need a real-world molecule, you can use a molecule generation tool or a database like ZINC to generate or retrieve a molecule that meets the specified criteria. 

Please let me know if you need any further assistance. 

Best regards, [Your Name] 

### Note:
The SMILES string provided is an example and
```

**QLoRA:**
```text
C1C2CC3CC1CC(C2)C3NC(=O)C1CCN(CC1)C(=O)C1CCN(CC1)C(=O)C1CCN(CC1)C(C1CCN(CC1)C(C1CCN(CC1)C(C1CCN(CC1)C(C1CCN(CC1)C)=O)=O)=O)=O
```

## Sample 11 (synllama)

**Instruction:** You are an expert synthetic organic chemist. Your task is to design a synthesis pathway for a given target molecule using common and reliable reaction templates and building blocks. Follow these instructions:

1. **Input the SMILES String:** Read in the SMILES string of the target molecule and identify common reaction templates that can be applied.

2. **Decompose the Target Molecule:** Use the identified reaction templates to decompose the target molecule into different intermediates.

3. **Check for Building Blocks:** For each intermediate:
   - Identify if it is a building block. If it is, wrap it in <bb> and </bb> tags and save it for later use.
   - If it is not a building block, apply additional reaction templates to further decompose it into building blocks.

4. **Document Reactions:** For each reaction documented in the output, wrap the reaction template in <rxn> and </rxn> tags.

5. **Repeat the Process:** Continue this process until all intermediates are decomposed into building blocks, and document each step clearly in a structured JSON format.

**Input:** Provide a synthetic pathway for this SMILES string: COc1cc2cc[nH]c2cc1-c1cc(N(C(=S)Nc2cccc(C)n2)C(C)c2ncccc2C(F)(F)F)ccc1OCCOC(=O)c1ccc2nc(C(NC(=O)OC(C)(C)C)C3CCCCC3)sc2c1

**Reference:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[N;$(N-[#6]):3]=[C;$(C=S):1].[N;$(N[#6]);!$(N=*);!$([N-]);!$(N#*);!$([ND3]);!$([ND4]);!$(N[O,N]);!$(N[C,S]=[S,O,N]):2]>>[N:3]-[C:1]-[N+0:2]</rxn>", "reactants": ["", "COc1cc2cc[nH]c2cc1-c1cc(NC(C)c2ncccc2C(F)(F)F)ccc1OCCOC(=O)c1ccc2nc(C(NC(=O)OC(C)(C)C)C3CCCCC3)sc2c1", "Cc1cccc(N=C=S)n1"], "product": "COc1cc2cc[nH]c2cc1-c1cc(N(C(=S)Nc2cccc(C)n2)C(C)c2ncccc2C(F)(F)F)ccc1OCCOC(=O)c1ccc2nc(C(NC(=O)OC(C)(C)C)C3CCCCC3)sc2c1"}, {"reaction_number": 2, "reaction_template": "<rxn>[Cl,Br,I][c;$(c1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):1].[N;$(NC)&!$(N=*)&!$([N-])&!$(N#*)&!$([ND3])&!$([ND4])&!$(N[c,O])&!$(N[C,S]=[S,O,N]),H2&$(Nc1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):2]>>[c:1][N:2]</rxn>", "reactants": ["", "COc1cc2cc[nH]c2cc1-c1cc(Cl)ccc1OCCOC(=O)c1ccc2nc(C(NC(=O)OC(C)(C)C)C3CCCCC3)sc2c1", "CC(N)c1ncccc1C(F)(F)F"], "product": "COc1cc2cc[nH]c2cc1-c1cc(NC(C)c2ncccc2C(F)(F)F)ccc1OCCOC(=O)c1ccc2nc(C(NC(=O)OC(C)(C)C)C3CCCCC3)sc2c1"}, {"reaction_number": 3, "reaction_template": "<rxn>[Br:1][c$(c(Br)),n$(n(Br)),o$(o(Br)),C$([CH](Br)(=C)):2].[C$(C(B)([CX4])([CX4])([CX4])),C$([CH](B)([CX4])([CX4])),C$([CH2](B)([CX4])),C$([CH2](B)),C$(C(B)(=C)),c$(c(B)),o$(o(B)),n$(n(B)):3][B$(B([C,c,n,o])([OH,$(OC)])([OH,$(OC)])),B$([B-1]([C,c,n,o])(N)([OH,$(OC)])([OH,$(OC)])):4]>>[C,c,n,o:2][C,c,n,o:3]</rxn>", "reactants": ["", "CC(C)(C)OC(=O)NC(c1nc2ccc(C(=O)OCCOc3ccc(Cl)cc3Br)cc2s1)C1CCCCC1", "COc1cc2cc[nH]c2cc1B1OC(C)(C)C(C)(C)O1"], "product": "COc1cc2cc[nH]c2cc1-c1cc(Cl)ccc1OCCOC(=O)c1ccc2nc(C(NC(=O)OC(C)(C)C)C3CCCCC3)sc2c1"}, {"reaction_number": 4, "reaction_template": "<rxn>[OH,O-]-[C$(C(=O)(O)([CX4,c])):2]=[O:3].[OH:8]-[C$([CH](O)([CX4,c])([CX4,c])),C$([CH2](O)([CX4,c])),C$([CH3](O)):6]>>[C:6][O]-[C:2]=[O:3]</rxn>", "reactants": ["", "CC(C)(C)OC(=O)NC(c1nc2ccc(C(=O)O)cc2s1)C1CCCCC1", "OCCOc1ccc(Cl)cc1Br"], "product": "CC(C)(C)OC(=O)NC(c1nc2ccc(C(=O)OCCOc3ccc(Cl)cc3Br)cc2s1)C1CCCCC1"}, {"reaction_number": 5, "reaction_template": "<rxn>[c;r6:1](-[SH1:2]):[c;r6:3](-[NH2:4]).[#6:6]-[CH1;R0:5](=[OD1])>>[c:3]2:[c:1]:[s:2]:[c:5](-[#6:6]):[n:4]2</rxn>", "reactants": ["", "CC(C)(C)OC(=O)NC(C=O)C1CCCCC1", "Nc1ccc(C(=O)O)cc1S"], "product": "CC(C)(C)OC(=O)NC(c1nc2ccc(C(=O)O)cc2s1)C1CCCCC1"}], "building_blocks": ["<bb>Cc1cccc(N=C=S)n1</bb>", "<bb>CC(N)c1ncccc1C(F)(F)F</bb>", "<bb>COc1cc2cc[nH]c2cc1B1OC(C)(C)C(C)(C)O1</bb>", "<bb>OCCOc1ccc(Cl)cc1Br</bb>", "<bb>Nc1ccc(C(=O)O)cc1S</bb>", "<bb>CC(C)(C)OC(=O)NC(C=O)C1CCCCC1</bb>"]}
```

**Base:**
```text
```json
{
  "synthesis_pathway": [
    {
      "target": "COc1cc2cc[nH]c2cc1-c1cc(N(C(=S)Nc2cccc(C)n2)C(C)c2ncccc2C(F)(F)F)ccc1OCCOC(=O)c1ccc2nc(C(NC(=O)OC(C)(C)C)C3CCCCC3)sc2c1",
      "intermediates": [
        {
          "name": "Intermediate 1",
          "smiles": "c1cc2cc[nH]c2cc1",
          "building_block": false,
          "reaction": {
            "template": "Friedel-Crafts Acylation",
            "reactants": ["benzene", "acetyl chloride"],
            "products": ["Intermediate 1"]
          }
        },
        {
          "name": "Intermediate 2",
          "smiles": "N(C(=S)Nc2cccc(C)n2)C(C)c2ncccc2C(F)(F)F",
          "building_block": false,
          "reaction": {
            "template": "Sulfur-Containing Building
```

**QLoRA:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[N;$(N-[#6]):3]=[C;$(C=S):1].[N;$(N[#6]);!$(N=*);!$([N-]);!$(N#*);!$([ND3]);!$([ND4]);!$(N[O,N]);!$(N[C,S]=[S,O,N]):2]>>[N:3]-[C:1]-[N+0:2]</rxn>", "reactants": ["", "COc1cc2cc[nH]c2cc1-c1cc(NC(C)c2ncccc2C(F)(F)F)ccc1OCCOC(=O)c1ccc2nc(C(NC(=O)OC(C)(C)C)C3CCCCC3)sc2c1", "Cc1cccc(N=C=S)n1"], "product": "COc1cc2cc[nH]c2cc1-c1cc(N(C(=S)Nc2cccc(C)n2)C(C)c2ncccc2C(F)(F)F)ccc1OCCOC(=O)c1ccc2nc
```

## Sample 12 (linkllama)

**Instruction:** You are an expert medicinal chemist specializing in linker design. Your task is to design a linker to connect given fragments and deduce whether the final molecule is chemically reasonable. The output should be in JSON format.

**Input:** Fragment 1 (SMILES: CNC(=O)[C@H](Cc1ccccc1)NC(=O)[*:1]) and Fragment 2 (SMILES: O=C1c2cccc3c(Br)ccc(c23)C(=O)N1CC[*:2]). The distance between the attachment points is 4.40 Angstroms, and the angle between them is 148.88 degrees. Given the above information about the fragments and attachment points, design a branched linker with >= 4 rotatable bonds to connect them. The final molecule should be unreasonable. And it should have the following properties: <= 5 H-bond donors, <= 200 TPSA.

**Reference:**
```text
{"linker": "CC(C)C[C@H](N[C@H](P(=O)(O)O)[*:2])[*:1]", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: phosphor."}
```

**Base:**
```text
```json
{
  "linker": {
    "smiles": "C1=CC=C(C=C1)C2=C(C=C(C=C2)C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=O)N)C(=O)N(C(=
```

**QLoRA:**
```text
{"linker": "CC(C)C[C@H](N[C@@H](C(=O)O)[*:2])[*:1]", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: phthalimide."}
```

## Sample 13 (linkllama)

**Instruction:** You are an expert medicinal chemist specializing in linker design. Your task is to design a linker to connect given fragments and deduce whether the final molecule is chemically reasonable. The output should be in JSON format.

**Input:** Fragment 1 (SMILES: CCCCCCCCCCCCCCC[*:1]) and Fragment 2 (SMILES: CC[n+]1ccccc1CN(C(C)=O)C(=O)O[*:2]). The distance between the attachment points is 8.44 Angstroms, and the angle between them is 109.33 degrees. Given the above information about the fragments and attachment points, design a linker with >= 7 heavy atoms to connect them. The final molecule should be unreasonable. And it should have the following properties: <= 1 H-bond donors, <= 10 H-bond acceptors, <= 600 Molecular weight.

**Reference:**
```text
{"linker": "C(OC[*:1])C1COC(C[*:2])C1", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: Aliphatic long chain."}
```

**Base:**
```text
```json
{
  "linker": {
    "smiles": "C1=CC=C(C=C1)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C)C(=O)N(C(C
```

**QLoRA:**
```text
{"linker": "C(C[*:1])OCC1COC(C[*:2])C1", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: Aliphatic long chain."}
```

## Sample 14 (synllama)

**Instruction:** You are an expert synthetic organic chemist. Your task is to design a synthesis pathway for a given target molecule using common and reliable reaction templates and building blocks. Follow these instructions:

1. **Input the SMILES String:** Read in the SMILES string of the target molecule and identify common reaction templates that can be applied.

2. **Decompose the Target Molecule:** Use the identified reaction templates to decompose the target molecule into different intermediates.

3. **Check for Building Blocks:** For each intermediate:
   - Identify if it is a building block. If it is, wrap it in <bb> and </bb> tags and save it for later use.
   - If it is not a building block, apply additional reaction templates to further decompose it into building blocks.

4. **Document Reactions:** For each reaction documented in the output, wrap the reaction template in <rxn> and </rxn> tags.

5. **Repeat the Process:** Continue this process until all intermediates are decomposed into building blocks, and document each step clearly in a structured JSON format.

**Input:** Provide a synthetic pathway for this SMILES string: CCOP(=O)(CC12CC(C(=O)C(CCOc3ccc(-c4nnn(C(c5ccccc5)c5ccnn5C)n4)cc3OC)N3N=NN=C3c3cccc(OCCCBr)c3)(C1)C2)OCC

**Reference:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[CH0;$(C-[#6]):1]#[NH0:2].[C;A;!$(C=O):3]-[*;#17,#35,#53]>>[C:1]1=[N:2]-N=N-N-1(-[C:3])</rxn>", "reactants": ["", "CCOP(=O)(CC12CC(C(=O)C(Cl)CCOc3ccc(-c4nnn(C(c5ccccc5)c5ccnn5C)n4)cc3OC)(C1)C2)OCC", "N#Cc1cccc(OCCCBr)c1"], "product": "CCOP(=O)(CC12CC(C(=O)C(CCOc3ccc(-c4nnn(C(c5ccccc5)c5ccnn5C)n4)cc3OC)N3N=NN=C3c3cccc(OCCCBr)c3)(C1)C2)OCC"}, {"reaction_number": 2, "reaction_template": "<rxn>[I:1][C$(C(I)([CX4,c])([CX4,c])([CX4,c])),C$([CH](I)([CX4,c])([CX4,c])),C$([CH2](I)([CX4,c])),C$([CH3](I)):2].[C$(C(=O)([Cl,OH,O-])([CX4,c])),C$([CH]([Cl,OH,O-])(=O)):3](=[O:6])[Cl,OH,O-:5]>>[C:2]-[C:3]=[O:6]</rxn>", "reactants": ["", "COc1cc(-c2nnn(C(c3ccccc3)c3ccnn3C)n2)ccc1OCCC(Cl)C(=O)O", "CCOP(=O)(CC12CC(I)(C1)C2)OCC"], "product": "CCOP(=O)(CC12CC(C(=O)C(Cl)CCOc3ccc(-c4nnn(C(c5ccccc5)c5ccnn5C)n4)cc3OC)(C1)C2)OCC"}, {"reaction_number": 3, "reaction_template": "<rxn>[C;H1&$(C([#6])[#6]),H2&$(C[#6]):1][OH1].[#7:2]1~[#7:3]~[#7H1:4]~[#7:5]~[#6:6]~1>>[#7:2]1:[#7:3]:[#7:4]([C:1]):[#7:5]:[#6:6]:1</rxn>", "reactants": ["", "COc1cc(-c2nn[nH]n2)ccc1OCCC(Cl)C(=O)O", "Cn1nccc1C(O)c1ccccc1"], "product": "COc1cc(-c2nnn(C(c3ccccc3)c3ccnn3C)n2)ccc1OCCC(Cl)C(=O)O"}, {"reaction_number": 4, "reaction_template": "<rxn>[OH+0,O-:5]-[C:3](=[O:4])-[C$([CH]([CX4])),C$([CH2]):2]>>[OH+0,O-:5]-[C:3](=[O:4])-[C:2]([Cl:6])</rxn>", "reactants": ["", "", "COc1cc(C2=NNN=N2)ccc1OCCCC(=O)O"], "product": "COc1cc(-c2nn[nH]n2)ccc1OCCC(Cl)C(=O)O"}, {"reaction_number": 5, "reaction_template": "<rxn>[CH0;$(C-[#6]):1]#[NH0:2]>>[C:1]1=[N:2]-N-N=N-1</rxn>", "reactants": ["", "", "COc1cc(C#N)ccc1OCCCC(=O)O"], "product": "COc1cc(C2=NNN=N2)ccc1OCCCC(=O)O"}], "building_blocks": ["<bb>N#Cc1cccc(OCCCBr)c1</bb>", "<bb>CCOP(=O)(CC12CC(I)(C1)C2)OCC</bb>", "<bb>Cn1nccc1C(O)c1ccccc1</bb>", "<bb>COc1cc(C#N)ccc1OCCCC(=O)O</bb>"]}
```

**Base:**
```text
```json
{
  "synthesis_pathway": [
    {
      "target": "CCOP(=O)(CC12CC(C(=O)C(CCOc3ccc(-c4nnn(C(c5ccccc5)c5ccnn5C)n4)cc3OC)N3N=NN=C3c3cccc(OCCCBr)c3)(C1)C2)OCC",
      "intermediates": [
        {
          "name": "CCOP(=O)(CC12CC(C(=O)C(CCOc3ccc(-c4nnn(C(c5ccccc5)c5ccnn5C)n4)cc3OC)N3N=NN=C3c3cccc(OCCCBr)c3)(C1)C2)OCC",
          "building_block": false,
          "reaction": {
            "template": "S_N2",
            "reactants": ["CCOP(=O)(CC12CC(C(=O)C(CCOc3ccc(-c4nnn(C(c5ccccc5)c5ccnn5C)n4)cc3OC)N3N=NN=C3c3cccc(OCCCBr)c3)(C1)C
```

**QLoRA:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[CH0;$(C-[#6]):1]#[NH0:2].[C;A;!$(C=O):3]-[*;#17,#35,#53]>>[C:1]1=[N:2]-N=N-N-1(-[C:3])</rxn>", "reactants": ["", "CCOP(=O)(CC12CC(C(=O)C(Br)CCOc3ccc(-c4nnn(C(c5ccccc5)c5ccnn5C)n4)cc3OC)(C1)C2)OCC", "N#Cc1cccc(OCCCBr)c1"], "product": "CCOP(=O)(CC12CC(C(=O)C(CCOc3ccc(-c4nnn(C(c5ccccc5)c5ccnn5C)n4)cc3OC)N3N=NN=C3c3cccc(OCCCBr)c3)(C1)C2)OCC"}, {"reaction_number": 2, "reaction_template": "<rxn>[I:1][C$(C(I)([CX4,c])(
```

## Sample 15 (smileyllama)

**Instruction:** You love and excel at generating SMILES strings of drug-like molecules

**Input:** Output a SMILES string for a drug-like molecule with the following properties: <= 7 Rotatable bonds, lacks covalent warheads, <= 3 LogP, <= 500 Molecular weight, <= 15 H-bond acceptors, a substructure of c1ccncc1:

**Reference:**
```text
c1cnc(cc1)-c1nn(c(N)n1)-c1c2c(cc(c(c2)[N+]([O-])=O)N2CCNCC2)ncn1
```

**Base:**
```text
Here is a SMILES string that meets the specified criteria:

CC(=O)Nc1ccccc1C(=O)N

This SMILES string represents a molecule with the following properties:

*   <= 7 Rotatable bonds: 6
*   Lacks covalent warheads
*   <= 3 LogP: 2.5
*   <= 500 Molecular weight: 246.26
*   <= 15 H-bond acceptors: 4
*   A substructure of c1ccncc1: present

This molecule meets all the specified criteria. However, please note that the actual properties of a molecule can vary based on the specific implementation of the calculation. The properties mentioned above are based on a general calculation and might not reflect the exact values obtained from a specific software or tool. 

The SMILES string provided is a representation of a molecule that meets the specified criteria. It is essential to verify the properties of the molecule using a reliable software or tool to ensure accuracy. 

If you need further assistance or have any questions, feel free to ask. I'll be happy to help. 

### Note:
The SMILES string provided is a representation of a molecule that meets the specified criteria. However,
```

**QLoRA:**
```text
c1c(ccc(c1)C(=O)N1CCN(CC1)C(=O)C1CCN(CC1)C(=O)c1ccncc1)F
```

## Sample 16 (linkllama)

**Instruction:** You are an expert medicinal chemist specializing in linker design. Your task is to design a linker to connect given fragments and deduce whether the final molecule is chemically reasonable. The output should be in JSON format.

**Input:** Fragment 1 (SMILES: CCc1cc(NC(=O)N[*:2])cc(-c2nnnn2C)c1) and Fragment 2 (SMILES: Fc1ccc(CC[*:1])cc1). The distance between the attachment points is 4.30 Angstroms, and the angle between them is 33.91 degrees. Given the above information about the fragments and attachment points, design a linker to connect them. The final molecule should be reasonable. And it should have the following properties: <= 3 H-bond donors, <= 10 H-bond acceptors, <= 140 TPSA.

**Reference:**
```text
{"linker": "C1C[C@@H](C[*:2])CN(C[*:1])C1", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: absent."}
```

**Base:**
```text
```json
{
  "linker": {
    "smiles": "C(=O)(N[*:1])C(=O)N[*:2]",
    "hbd": 1,
    "hba": 2,
    "tpsa": 45.6
  },
  "final_molecule": {
    "smiles": "CCc1cc(NC(=O)N[*:2])cc(-c2nnnn2C)c1C(=O)(N[*:1])C(=O)N[*:2]Fc1ccc(CC[*:1])cc1",
    "hbd": 2,
    "hba": 12,
    "tpsa": 145.6
  },
  "reasonability": {
    "hbd": "The final molecule has 2 H-bond donors, which is <= 3.",
    "hba": "The final molecule has 12 H-bond acceptors, which is <= 10.",
    "tpsa": "The final molecule has a TPSA of 145.6, which is <= 140."
  }
}
```
### Explanation:
To design a linker, we need to consider the distance and angle
```

**QLoRA:**
```text
{"linker": "C1C[C@H](C[*:2])CN([*:1])C1", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: absent."}
```

## Sample 17 (smileyllama)

**Instruction:** You love and excel at generating SMILES strings of drug-like molecules

**Input:** Output a SMILES string for a drug-like molecule with the following properties: <= 600 Molecular weight, <= 200 TPSA, <= 10 H-bond acceptors, <= 7 H-bond donors, <= 10 Rotatable bonds, lacks covalent warheads:

**Reference:**
```text
c1c(F)ccc(N2CC3=C(N(C2c2ccc(cc2)Br)c2ccccc2)CC(C)(C)CC3=O)c1
```

**Base:**
```text
Here is a SMILES string for a drug-like molecule that meets the specified properties:

`CC(=O)Nc1ccc(cc1)C(=O)N`

This SMILES string represents a molecule with the following properties:

*   Molecular weight: 246.28
*   TPSA: 104.45
*   H-bond acceptors: 4
*   H-bond donors: 1
*   Rotatable bonds: 4
*   Covalent warheads: None

This molecule meets the specified criteria and is a valid SMILES string. However, please note that the properties of the molecule can be adjusted by modifying the SMILES string. The properties mentioned above are based on the provided SMILES string and may not reflect the actual properties of the molecule in a real-world scenario. 

The SMILES string `CC(=O)Nc1ccc(cc1)C(=O)N` represents a molecule with a molecular weight of 246.28, which is less than or equal to 600. The TPSA (Topological Polar Surface Area) is 104.45, which is less than or equal to 200. The molecule has 4 H-bond acceptors and 1
```

**QLoRA:**
```text
c1c(ccc(c1)C(=O)Nc1cc2c(cc1)nc(N1CCN(CC1)C(=O)C1CC1)[nH]2)C
```

## Sample 18 (linkllama)

**Instruction:** You are an expert medicinal chemist specializing in linker design. Your task is to design a linker to connect given fragments and deduce whether the final molecule is chemically reasonable. The output should be in JSON format.

**Input:** Fragment 1 (SMILES: CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)[*:2])C(=O)N[C@@H](CC(C)C)C(=O)O) and Fragment 2 (SMILES: C[N+](C)(C)CCCC[C@H](N)[*:1]). The distance between the attachment points is 7.49 Angstroms, and the angle between them is 89.97 degrees. Given the above information about the fragments and attachment points, design a branched linker with >= 1 rotatable bonds to connect them. The final molecule should be unreasonable. And it should have the following properties: <= 10 H-bond acceptors, > 700 Molecular weight.

**Reference:**
```text
{"linker": "N=C(NCCC[*:2])NC(=O)[*:1]", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: Aliphatic long chain."}
```

**Base:**
```text
```json
{
  "linker": {
    "smiles": "C1=CC(=C(C=C1)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C
```

**QLoRA:**
```text
{"linker": "N=C(NCCC[*:2])NC(=O)[*:1]", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: pass. REOS failed rule: Aliphatic long chain."}
```

## Sample 19 (linkllama)

**Instruction:** You are an expert medicinal chemist specializing in linker design. Your task is to design a linker to connect given fragments and deduce whether the final molecule is chemically reasonable. The output should be in JSON format.

**Input:** Fragment 1 (SMILES: CC[C@@]1(O)C[C@H](O[C@@H]2CC(N(C)C)[C@@H](O[*:1])[C@@H](C)O2)c2c(cc3c(c2O)C(=O)c2c(O)ccc(O)c2C3=O)[C@H]1C(=O)OC) and Fragment 2 (SMILES: C[C@H]1O[C@H](O[*:2])CCC1=O). The distance between the attachment points is 5.52 Angstroms, and the angle between them is 175.96 degrees. Given the above information about the fragments and attachment points, design a ring-containing linker with 0 rotatable bonds, >= 7 heavy atoms to connect them. The final molecule should be unreasonable. And it should have the following properties: <= 5 H-bond donors, > 10 H-bond acceptors, > 700 Molecular weight.

**Reference:**
```text
{"linker": "CC1O[C@H]([*:1])C[C@@H](O)[C@H]1[*:2]", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: fail. REOS failed rule: hydroquinone."}
```

**Base:**
```text
```json
{
  "linker": {
    "smiles": "C1=CC(=O)N(C)C2=C(C(=O)C3=C(C(=O)C4=C(C(=O)C5=C(C(=O)C6=C(C(=O)C7=C(C(=O)C8=C(C(=O)C9=C(C(=O)C10=C(C(=O)C11=C(C(=O)C12=C(C(=O)C13=C(C(=O)C14=C(C(=O)C15=C(C(=O)C16=C(C(=O)C17=C(C(=O)C18=C(C(=O)C19=C(C(=O)C20=C(C(=O)C21=C(C(=O)C22=C(C(=O)C23=C(C(=O)C24=C(C(=O)C25=C(C(=O)C26=C(C(=O)C27=C(C(=O)C28=C(C(=O)C29=C(C(=O)C30=C(C(=O)C31=C(C(=O)C32=C(C(=O)C33=C(C(=O)C34=C(C(=O
```

**QLoRA:**
```text
{"linker": "C[C@H]1O[C@@H]([*:1])C[C@H](O)[C@@H]1[*:2]", "reasoning": "Linker bad rings: pass. Linker problematic ring: absent. Undesirable SMARTS: pass. PAINS: fail. REOS failed rule: hydroquinone."}
```

## Sample 20 (synllama)

**Instruction:** You are an expert synthetic organic chemist. Your task is to design a synthesis pathway for a given target molecule using common and reliable reaction templates and building blocks. Follow these instructions:

1. **Input the SMILES String:** Read in the SMILES string of the target molecule and identify common reaction templates that can be applied.

2. **Decompose the Target Molecule:** Use the identified reaction templates to decompose the target molecule into different intermediates.

3. **Check for Building Blocks:** For each intermediate:
   - Identify if it is a building block. If it is, wrap it in <bb> and </bb> tags and save it for later use.
   - If it is not a building block, apply additional reaction templates to further decompose it into building blocks.

4. **Document Reactions:** For each reaction documented in the output, wrap the reaction template in <rxn> and </rxn> tags.

5. **Repeat the Process:** Continue this process until all intermediates are decomposed into building blocks, and document each step clearly in a structured JSON format.

**Input:** Provide a synthetic pathway for this SMILES string: COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2(Cl)c1cc(OC(F)(F)F)cc(-c2cnnn2CCCCCNC(=O)N2CCOCC2)c1NC(=O)c1cc2c(o1)CCCN(C(=O)OC(C)(C)C)C2

**Reference:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[Cl,OH,O-:3][C$(C(=O)([CX4,c])),C$([CH](=O)):2]=[O:4].[N$([NH2,NH3+1]([CX4,c])),N$([NH]([CX4,c])([CX4,c])):6]>>[N+0:6]-[C:2]=[O:4]</rxn>", "reactants": ["", "COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2(Cl)c1cc(OC(F)(F)F)cc(-c2cnnn2CCCCCNC(=O)N2CCOCC2)c1N", "CC(C)(C)OC(=O)N1CCCc2oc(C(=O)O)cc2C1"], "product": "COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2(Cl)c1cc(OC(F)(F)F)cc(-c2cnnn2CCCCCNC(=O)N2CCOCC2)c1NC(=O)c1cc2c(o1)CCCN(C(=O)OC(C)(C)C)C2"}, {"reaction_number": 2, "reaction_template": "<rxn>[OH:2]-[$([CX4]),c:1]>>[Cl:3][C,c:1]</rxn>", "reactants": ["", "", "COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2(O)c1cc(OC(F)(F)F)cc(C2=CN=NN2CCCCCNC(=O)N2CCOCC2)c1N"], "product": "COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2(Cl)c1cc(OC(F)(F)F)cc(-c2cnnn2CCCCCNC(=O)N2CCOCC2)c1N"}, {"reaction_number": 3, "reaction_template": "<rxn>[CH0;$(C-[#6]):1]#[CH1:2].[C;H1,H2;A;!$(C=O):3]-[*;#17,#35,#53,OH1]>>[C:1]1=[C:2]-N=NN(-[C:3])-1</rxn>", "reactants": ["", "C#Cc1cc(OC(F)(F)F)cc(C2(O)C3(C(=O)OC)CCC(=O)C2(C(=O)OC)C3)c1N", "O=C(NCCCCCO)N1CCOCC1"], "product": "COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2(O)c1cc(OC(F)(F)F)cc(C2=CN=NN2CCCCCNC(=O)N2CCOCC2)c1N"}, {"reaction_number": 4, "reaction_template": "<rxn>[#6:1][C;H1,$([C]([#6])[#6]):2]=[OD1:3].[Cl,Br,I][#6;$([#6]~[#6]);!$([#6]([Cl,Br,I])[Cl,Br,I]);!$([#6]=O):4]>>[C:1][#6:2]([OH1:3])[#6:4]</rxn>", "reactants": ["", "COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2=O", "C#Cc1cc(OC(F)(F)F)cc(Br)c1N"], "product": "C#Cc1cc(OC(F)(F)F)cc(C2(O)C3(C(=O)OC)CCC(=O)C2(C(=O)OC)C3)c1N"}, {"reaction_number": 5, "reaction_template": "<rxn>[O$(O(C)([CX4])):8][C:7](=[O:9])[CH:6][C:5][C:11][C:4][C:3][C:2]([O$(O(C)([CX4])):10])=[O:1]>>[O:8][C:7](=[O:9])[C:6]1[C:5][C:11][C:4][C:3][C:2]1=[O:1]</rxn>", "reactants": ["", "", "COC(=O)C1CC(C(=O)OC)(C(=O)OC)CCC1=O"], "product": "COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2=O"}], "building_blocks": ["<bb>CC(C)(C)OC(=O)N1CCCc2oc(C(=O)O)cc2C1</bb>", "<bb>O=C(NCCCCCO)N1CCOCC1</bb>", "<bb>C#Cc1cc(OC(F)(F)F)cc(Br)c1N</bb>", "<bb>COC(=O)C1CC(C(=O)OC)(C(=O)OC)CCC1=O</bb>"]}
```

**Base:**
```text
```json
{
  "synthesis_pathway": [
    {
      "target": "COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2(Cl)c1cc(OC(F)(F)F)cc(-c2cnnn2CCCCCNC(=O)N2CCOCC2)c1NC(=O)c1cc2c(o1)CCCN(C(=O)OC(C)(C)C)C2",
      "intermediates": [
        {
          "name": "Intermediate 1",
          "smiles": "COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2(Cl)",
          "building_block": false,
          "reaction": {
            "template": "<rxn>COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2(Cl) -> C12CCC(=O)C(C(=O)OC)(C1)C2(Cl)</rxn>",
            "conditions": "Heating in a solvent"
          },
          "further_decomposition": [
            {
              "name": "Intermediate 2",
              "smiles
```

**QLoRA:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[OH:2]-[$([CX4]),c:1]>>[Cl:3][C,c:1]</rxn>", "reactants": ["", "", "COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2(O)c1cc(OC(F)(F)F)cc(-c2cnnn2CCCCCNC(=O)N2CCOCC2)c1NC(=O)c1cc2c(o1)CCCN(C(=O)OC(C)(C)C)C2"], "product": "COC(=O)C12CCC(=O)C(C(=O)OC)(C1)C2(Cl)c1cc(OC(F)(F)F)cc(-c2cnnn2CCCCCNC(=O)N2CCOCC2)c1NC(=O)c1cc2c(o1)CCCN(C(=O)OC(C)(C)C)C2"}, {"reaction_number": 2, "reaction_template": "<rxn>[#6:1][C;H1,$([C]([#
```

## Sample 21 (smileyllama)

**Instruction:** You love and excel at generating SMILES strings of drug-like molecules

**Input:** Output a SMILES string for a drug-like molecule with the following properties: <= 10 Rotatable bonds, <= 6 LogP, <= 5 H-bond acceptors, lacks bad SMARTS:

**Reference:**
```text
C(N(c1c(C)cc(cc1)C)CC(NCc1occc1)=O)(=O)CNS(=O)(=O)c1ccccc1
```

**Base:**
```text
Here is a SMILES string for a drug-like molecule that meets the specified properties:

CC(=O)Nc1ccc(cc1)C(=O)N

This SMILES string represents a molecule with the following properties:

*   <= 10 Rotatable bonds: 4
*   <= 6 LogP: 2.5
*   <= 5 H-bond acceptors: 3
*   Lacks bad SMARTS: Yes

This molecule is a drug-like compound with a relatively simple structure, making it suitable for further analysis or modification. The SMILES string can be used as a starting point for generating 3D models, calculating properties, or performing other computational tasks. 

Note: The properties mentioned above are based on the given SMILES string and may not reflect the actual properties of the molecule in real-world scenarios. For accurate predictions, it is recommended to use a reliable molecular modeling software or service. 

### Additional Information:
The SMILES string provided is a simplified representation of a molecule and may not be a real-world compound. It is intended for illustrative purposes only. If you need to generate a SMILES string for a specific molecule or property set, please provide more detailed information about the desired molecule, such as its
```

**QLoRA:**
```text
c1c(ccc(c1)C(=O)Nc1ccccc1)C(Nc1ccccc1)=O
```

## Sample 22 (synllama)

**Instruction:** You are an expert synthetic organic chemist. Your task is to design a synthesis pathway for a given target molecule using common and reliable reaction templates and building blocks. Follow these instructions:

1. **Input the SMILES String:** Read in the SMILES string of the target molecule and identify common reaction templates that can be applied.

2. **Decompose the Target Molecule:** Use the identified reaction templates to decompose the target molecule into different intermediates.

3. **Check for Building Blocks:** For each intermediate:
   - Identify if it is a building block. If it is, wrap it in <bb> and </bb> tags and save it for later use.
   - If it is not a building block, apply additional reaction templates to further decompose it into building blocks.

4. **Document Reactions:** For each reaction documented in the output, wrap the reaction template in <rxn> and </rxn> tags.

5. **Repeat the Process:** Continue this process until all intermediates are decomposed into building blocks, and document each step clearly in a structured JSON format.

**Input:** Provide a synthetic pathway for this SMILES string: Cc1cc(Oc2ccc(-c3n[nH]c(C4(Cc5cnnn5Cc5ccccc5N5CCCC5)CCCCO4)n3)cc2)ccc1NC(=S)Nc1ccc(-c2nc(-c3cncc(CC(=O)OC4(c5cccc(C(=O)O)c5)COC4)c3)n[nH]2)cn1

**Reference:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[Cl,OH,O-:3][C$(C(=O)([CX4,c])),C$([CH](=O)):2]=[O:4].[O$([OH]([CX4,c])),O$([OH]([CX4,c])([CX4,c])),S$([SH]([CX4,c])),S$([SH]([CX4,c])([CX4,c])):6]>>[*:6]-[C:2]=[O:4]</rxn>", "reactants": ["", "Cc1cc(Oc2ccc(-c3n[nH]c(C4(Cc5cnnn5Cc5ccccc5N5CCCC5)CCCCO4)n3)cc2)ccc1NC(=S)Nc1ccc(C2=NC(c3cncc(CC(=O)O)c3)=NN2)cn1", "O=C(O)c1cccc(C2(O)COC2)c1"], "product": "Cc1cc(Oc2ccc(-c3n[nH]c(C4(Cc5cnnn5Cc5ccccc5N5CCCC5)CCCCO4)n3)cc2)ccc1NC(=S)Nc1ccc(-c2nc(-c3cncc(CC(=O)OC4(c5cccc(C(=O)O)c5)COC4)c3)n[nH]2)cn1"}, {"reaction_number": 2, "reaction_template": "<rxn>[CH0;$(C-[#6]):1]#[NH0:2].[CH0;$(C-[#6]);R0:5](=[OD1])-[#8;H1,$(O-[CH3]),$(O-[CH2]-[CH3])]>>[N:2]1-[C:1]=N-N-[C:5]=1</rxn>", "reactants": ["", "COC(=O)c1ccc(NC(=S)Nc2ccc(Oc3ccc(-c4n[nH]c(C5(Cc6cnnn6Cc6ccccc6N6CCCC6)CCCCO5)n4)cc3)cc2C)nc1", "N#Cc1cncc(CC(=O)O)c1"], "product": "Cc1cc(Oc2ccc(-c3n[nH]c(C4(Cc5cnnn5Cc5ccccc5N5CCCC5)CCCCO4)n3)cc2)ccc1NC(=S)Nc1ccc(C2=NC(c3cncc(CC(=O)O)c3)=NN2)cn1"}, {"reaction_number": 3, "reaction_template": "<rxn>[N;$(N-[#6]):3]=[C;$(C=S):1].[N;$(N[#6]);!$(N=*);!$([N-]);!$(N#*);!$([ND3]);!$([ND4]);!$(N[O,N]);!$(N[C,S]=[S,O,N]):2]>>[N:3]-[C:1]-[N+0:2]</rxn>", "reactants": ["", "Cc1cc(Oc2ccc(C3=NNC(C4(Cc5cnnn5Cc5ccccc5N5CCCC5)CCCCO4)=N3)cc2)ccc1N", "COC(=O)c1ccc(N=C=S)nc1"], "product": "COC(=O)c1ccc(NC(=S)Nc2ccc(Oc3ccc(-c4n[nH]c(C5(Cc6cnnn6Cc6ccccc6N6CCCC6)CCCCO5)n4)cc3)cc2C)nc1"}, {"reaction_number": 4, "reaction_template": "<rxn>[CH0;$(C-[#6]):1]#[NH0:2].[CH0;$(C-[#6]);R0:5](=[OD1])-[#8;H1,$(O-[CH3]),$(O-[CH2]-[CH3])]>>[N:2]1-[C:1]=N-N-[C:5]=1</rxn>", "reactants": ["", "CCOC(=O)C1(CC2=CN=NN2Cc2ccccc2N2CCCC2)CCCCO1", "Cc1cc(Oc2ccc(C#N)cc2)ccc1N"], "product": "Cc1cc(Oc2ccc(C3=NNC(C4(Cc5cnnn5Cc5ccccc5N5CCCC5)CCCCO4)=N3)cc2)ccc1N"}, {"reaction_number": 5, "reaction_template": "<rxn>[CH0;$(C-[#6]):1]#[CH1:2].[C;H1,H2;A;!$(C=O):3]-[*;#17,#35,#53,OH1]>>[C:1]1=[C:2]-N=NN(-[C:3])-1</rxn>", "reactants": ["", "C#CCC1(C(=O)OCC)CCCCO1", "OCc1ccccc1N1CCCC1"], "product": "CCOC(=O)C1(CC2=CN=NN2Cc2ccccc2N2CCCC2)CCCCO1"}], "building_blocks": ["<bb>O=C(O)c1cccc(C2(O)COC2)c1</bb>", "<bb>N#Cc1cncc(CC(=O)O)c1</bb>", "<bb>COC(=O)c1ccc(N=C=S)nc1</bb>", "<bb>Cc1cc(Oc2ccc(C#N)cc2)ccc1N</bb>", "<bb>OCc1ccccc1N1CCCC1</bb>", "<bb>C#CCC1(C(=O)OCC)CCCCO1</bb>"]}
```

**Base:**
```text
```json
{
  "synthetic_pathway": [
    {
      "target": "Cc1cc(Oc2ccc(-c3n[nH]c(C4(Cc5cnnn5Cc5ccccc5N5CCCC5)CCCCO4)n3)cc2)ccc1NC(=S)Nc1ccc(-c2nc(-c3cncc(CC(=O)OC4(c5cccc(C(=O)O)c5)COC4)c3)n[nH]2)cn1",
      "intermediates": [
        {
          "name": "Intermediate 1",
          "smiles": "Cc1cc(Oc2ccc(-c3n[nH]c(C4(Cc5cnnn5Cc5ccccc5N5CCCC5)CCCCO4)n3)cc2)ccc1",
          "building_block": false,
          "reaction": {
            "template": "Ullmann Coupling",
            "reactants": ["C1CC(=O)O", "C2CC(=O)O"],
            "products": ["C1CC(=O)OC2CC(=O)O"]
          }
        },
        {
          "name
```

**QLoRA:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[Cl,OH,O-:3][C$(C(=O)([CX4,c])),C$([CH](=O)):2]=[O:4].[O$([OH]([CX4,c])),O$([OH]([CX4,c])([CX4,c])),S$([SH]([CX4,c])),S$([SH]([CX4,c])([CX4,c])):6]>>[*:6]-[C:2]=[O:4]</rxn>", "reactants": ["", "Cc1cc(Oc2ccc(-c3n[nH]c(C4(Cc5cnnn5Cc5ccccc5N5CCCC5)CCCCO4)n3)cc2)ccc1NC(=S)Nc1ccc(C2=NC(c3cncc(CC(=O)O)c3)=NN2)cn1", "O=C(O)c1cccc(C2(O)COC2)c1"], "product": "Cc1cc(Oc2ccc(-c3n[nH]c(C4(Cc5cnnn5Cc5ccccc5N5
```

## Sample 23 (synllama)

**Instruction:** You are an expert synthetic organic chemist. Your task is to design a synthesis pathway for a given target molecule using common and reliable reaction templates and building blocks. Follow these instructions:

1. **Input the SMILES String:** Read in the SMILES string of the target molecule and identify common reaction templates that can be applied.

2. **Decompose the Target Molecule:** Use the identified reaction templates to decompose the target molecule into different intermediates.

3. **Check for Building Blocks:** For each intermediate:
   - Identify if it is a building block. If it is, wrap it in <bb> and </bb> tags and save it for later use.
   - If it is not a building block, apply additional reaction templates to further decompose it into building blocks.

4. **Document Reactions:** For each reaction documented in the output, wrap the reaction template in <rxn> and </rxn> tags.

5. **Repeat the Process:** Continue this process until all intermediates are decomposed into building blocks, and document each step clearly in a structured JSON format.

**Input:** Provide a synthetic pathway for this SMILES string: CCN(Cc1ccc(F)c(CN)c1)S(=O)(=O)c1cc([N+](=O)[O-])ccc1-c1ccc2nc(CCc3nnn(C(Cc4nc(C)c[nH]4)C4(NC(=O)OC(C)(C)C)CC5(CCCCC5)C4)n3)n(C(=O)c3ccco3)c2c1

**Reference:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[C;H1&$(C([#6])[#6]),H2&$(C[#6]):1][OH1].[NH1;$(N([#6])S(=O)=O):2]>>[C:1][N:2]</rxn>", "reactants": ["", "CCNS(=O)(=O)c1cc([N+](=O)[O-])ccc1c1ccc2nc(CCc3nnn(C(Cc4nc(C)c[nH]4)C4(NC(=O)OC(C)(C)C)CC5(CCCCC5)C4)n3)n(C(=O)c3ccco3)c2c1", "NCc1cc(CO)ccc1F"], "product": "CCN(Cc1ccc(F)c(CN)c1)S(=O)(=O)c1cc([N+](=O)[O-])ccc1-c1ccc2nc(CCc3nnn(C(Cc4nc(C)c[nH]4)C4(NC(=O)OC(C)(C)C)CC5(CCCCC5)C4)n3)n(C(=O)c3ccco3)c2c1"}, {"reaction_number": 2, "reaction_template": "<rxn>[c;$(c1[c;$(c[C,S,N](=[OD1])[*;R0;!OH1])]cccc1):1][C;$(C(=O)[O;H1])].[c;$(c1aaccc1):2][Cl,Br,I]>>[c:1][c:2]</rxn>", "reactants": ["", "Cc1c[nH]c(CC(n2nnc(CCc3nc4ccc(Cl)cc4n3C(=O)c3ccco3)n2)C2(NC(=O)OC(C)(C)C)CC3(CCCCC3)C2)n1", "CCNS(=O)(=O)c1cc([N+](=O)[O-])ccc1C(=O)O"], "product": "CCNS(=O)(=O)c1cc([N+](=O)[O-])ccc1c1ccc2nc(CCc3nnn(C(Cc4nc(C)c[nH]4)C4(NC(=O)OC(C)(C)C)CC5(CCCCC5)C4)n3)n(C(=O)c3ccco3)c2c1"}, {"reaction_number": 3, "reaction_template": "<rxn>[c;r6:1](-[NH1;$(N-[#6]):2]):[c;r6:3](-[NH2:4]).[#6:6]-[C;R0:5](=[OD1])-[#8;H1,$(O-[CH3])]>>[c:3]2:[c:1]:[n:2]:[c:5](-[#6:6]):[n:4]2</rxn>", "reactants": ["", "Cc1c[nH]c(CC(n2nnc(CCC(=O)O)n2)C2(NC(=O)OC(C)(C)C)CC3(CCCCC3)C2)n1", "Nc1ccc(Cl)cc1NC(=O)c1ccco1"], "product": "Cc1c[nH]c(CC(n2nnc(CCc3nc4ccc(Cl)cc4n3C(=O)c3ccco3)n2)C2(NC(=O)OC(C)(C)C)CC3(CCCCC3)C2)n1"}, {"reaction_number": 4, "reaction_template": "<rxn>[C;H1&$(C([#6])[#6]),H2&$(C[#6]):1][OH1].[#7:2]1~[#7:3]~[#7H1:4]~[#7:5]~[#6:6]~1>>[#7:2]1:[#7:3]:[#7:4]([C:1]):[#7:5]:[#6:6]:1</rxn>", "reactants": ["", "Cc1c[nH]c(CC(O)C2(NC(=O)OC(C)(C)C)CC3(CCCCC3)C2)n1", "O=C(O)CCc1nn[nH]n1"], "product": "Cc1c[nH]c(CC(n2nnc(CCC(=O)O)n2)C2(NC(=O)OC(C)(C)C)CC3(CCCCC3)C2)n1"}, {"reaction_number": 5, "reaction_template": "<rxn>[#6:1][C;H1,$([C]([#6])[#6]):2]=[OD1:3].[Cl,Br,I][#6;$([#6]~[#6]);!$([#6]([Cl,Br,I])[Cl,Br,I]);!$([#6]=O):4]>>[C:1][#6:2]([OH1:3])[#6:4]</rxn>", "reactants": ["", "Cc1c[nH]c(CCl)n1", "CC(C)(C)OC(=O)NC1(C=O)CC2(CCCCC2)C1"], "product": "Cc1c[nH]c(CC(O)C2(NC(=O)OC(C)(C)C)CC3(CCCCC3)C2)n1"}], "building_blocks": ["<bb>NCc1cc(CO)ccc1F</bb>", "<bb>CCNS(=O)(=O)c1cc([N+](=O)[O-])ccc1C(=O)O</bb>", "<bb>Nc1ccc(Cl)cc1NC(=O)c1ccco1</bb>", "<bb>O=C(O)CCc1nn[nH]n1</bb>", "<bb>CC(C)(C)OC(=O)NC1(C=O)CC2(CCCCC2)C1</bb>", "<bb>Cc1c[nH]c(CCl)n1</bb>"]}
```

**Base:**
```text
```json
{
  "synthetic_pathway": [
    {
      "target": "CCN(Cc1ccc(F)c(CN)c1)S(=O)(=O)c1cc([N+](=O)[O-])ccc1-c1ccc2nc(CCc3nnn(C(Cc4nc(C)c[nH]4)C4(NC(=O)OC(C)(C)C)CC5(CCCCC5)C4)n3)n(C(=O)c3ccco3)c2c1)c1",
      "intermediates": [
        {
          "name": "CCN(Cc1ccc(F)c(CN)c1)",
          "building_block": false,
          "reaction": {
            "template": "Ar-arylation",
            "reactants": ["CCN(Cc1ccc(F)c(CN)c1)", "c1ccc(F)c(CN)c1"],
            "products": ["CCN(Cc1ccc(F)c(CN)c1)"]
          }
        },
        {
          "name": "S(=O)(=O)c1cc([N+](=O)[O-])ccc1-c1ccc2nc(CCc3nnn(C
```

**QLoRA:**
```text
{"reactions": [{"reaction_number": 1, "reaction_template": "<rxn>[C;H1&$(C([#6])[#6]),H2&$(C[#6]):1][OH1].[NH1;$(N([#6])S(=O)=O):2]>>[C:1][N:2]</rxn>", "reactants": ["", "CCNS(=O)(=O)c1cc([N+](=O)[O-])ccc1c1ccc2nc(CCc3nnn(C(Cc4nc(C)c[nH]4)C4(NC(=O)OC(C)(C)C)CC5(CCCCC5)C4)n3)n(C(=O)c3ccco3)c2c1", "NCc1cc(CO)ccc1F"], "product": "CCN(Cc1ccc(F)c(CN)c1)S(=O)(=O)c1cc([N+](=O)[O-])ccc1-c1ccc2nc(CCc3nnn(C(Cc4nc(C)c[nH]4)C4(NC(=O)OC(C)(C)C)CC5(CCCCC5
```

## Sample 24 (smileyllama)

**Instruction:** You love and excel at generating SMILES strings of drug-like molecules

**Input:** Output a SMILES string for a drug-like molecule with the following properties: A formula of C16H25N3O4S, <= 6 LogP, a substructure of CC(C)NC(N)=O:

**Reference:**
```text
c1cc(OCCN2CCCC2)ccc1S(NC(NC(C)C)=O)(=O)=O
```

**Base:**
```text
CC(C)N(C(=O)Nc1ccc(cc1)S(=O)(=O)N)C(=O)N

This SMILES string represents a drug-like molecule with the specified properties. The molecule has a formula of C16H25N3O4S, a LogP of 4.5 (which is <= 6), and contains the substructure CC(C)NC(N)=O. 

Note: The SMILES string is a concise way to represent a molecule using a sequence of atoms and bonds. It is a widely used format in chemistry and is often used in computational chemistry and cheminformatics. 

The SMILES string provided above represents a molecule with the specified properties. The molecule has a formula of C16H25N3O4S, which matches the given formula. The LogP of 4.5 is within the specified range of <= 6. The substructure CC(C)NC(N)=O is also present in the molecule, as indicated by the SMILES string. 

Please note that the SMILES string is a representation of a molecule and not the molecule itself. The actual molecule may have different properties and structures depending on the specific conditions and environment. 

Also, note that the SMILES string
```

**QLoRA:**
```text
C1C(CCC1)C(NC(=O)NC(C)C)C(=O)Nc1ccc(cc1)S(=O)(C)=O
```

## Sample 25 (smileyllama)

**Instruction:** You love and excel at generating SMILES strings of drug-like molecules

**Input:** Output a SMILES string for a drug-like molecule with the following properties: lacks covalent warheads, a substructure of O=CCn1c(=O)oc2ccccc21, A formula of C21H18N4O5, <= 10 Rotatable bonds, <= 15 H-bond acceptors, <= 0.4 Fraction sp3, <= 3 H-bond donors, lacks bad SMARTS:

**Reference:**
```text
N1C(=O)C2(N(C)C1=O)Cc1ccc(NC(=O)Cn3c4ccccc4oc3=O)cc1C2
```

**Base:**
```text
CC(=O)Nc1ccc(cc1)C(=O)Nc2ccc(cc2)C(=O)Nc3ccc(cc3)C(=O)N

### Explanation:
The SMILES string provided represents a drug-like molecule with the specified properties. It includes a substructure of O=CCn1c(=O)oc2ccccc21, which matches the given substructure. The formula C21H18N4O5 and the specified properties (<= 10 Rotatable bonds, <= 15 H-bond acceptors, <= 0.4 Fraction sp3, <= 3 H-bond donors) are also satisfied. The SMILES string does not contain any covalent warheads or bad SMARTS. 

### Note:
The SMILES string is a concise way to represent a molecule using a line notation. The properties mentioned in the task (e.g., Rotatable bonds, H-bond acceptors, Fraction sp3, H-bond donors) are calculated based on the molecular structure and are used to evaluate the drug-likeness of the molecule. The substructure O=CCn1c(=O)oc2ccccc21 is a part of the molecule's structure, and the formula C
```

**QLoRA:**
```text
c1c2c(ccc1)oc(n2CC(Nc1ccc(cc1)NC(=O)Nc1ccccc1OC)=O)=O
```

