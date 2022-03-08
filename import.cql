LOAD CSV with headers FROM 'file:///researchers_cleaned.csv' AS row
WITH
    row,
    split(row.alma, ",") AS almas
//    split(row.field, ",") AS fields
//    split(row.award, ",") AS awards
UNWIND almas AS alma
//UNWIND fields as field
//UNWIND awards as award
MERGE (r:Researcher {researcherId: row.id, name:row.label})
MERGE (s:School {name: alma})
//MERGE (f:Field {name: field})
//MERGE (a:Award {name: award})
MERGE (r)-[:EDUCATED_AT]->(s)
//MERGE (r)-[:STUDIED]->(f)
//MERGE (r)-[:RECEIVED]->(a)
