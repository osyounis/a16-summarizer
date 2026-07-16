# Qualitative examples — base vs. fine-tuned

Five dialogues at fixed quantiles of the per-example ROUGE-L delta (tuned − base), ranked ascending. Deterministic and unbiased by construction: the first example is where the fine-tune did **worst** relative to base.

---

## 0% (worst for tuned) — test row 357

ROUGE-L: base 0.5294 | tuned 0.2609 | Δ -0.2685

### Dialogue

```
#Person1#: Hi, can I help you?
#Person2#: Yes. We have a reservation.
#Person1#: What's your name, Sir?
#Person2#: Jack.
#Person1#: Yes Sir. We have your reservation. Could you fill up these forms, please?
#Person2#: The reservation was for a double, for three nights. Wasn't it?
#Person1#: Yes, Sir. That's correct.
```

### Human references (3)

1. #Person1# confirms Jack's reservation for a double for three nights. Jack checks in.
2. Jack checks in at #Person1#'s hotel.
3. Jack has a hotel reservation and #Person1# helps him check in.

### Base output

> #Person1# greets #Person2# and confirms their reservation details. #Person2# confirms the reservation was for a double room for three nights.

### Fine-tuned output

> #Person1# helps Jack to fill up the reservation forms.

---

## 25% — test row 484

ROUGE-L: base 0.3043 | tuned 0.3818 | Δ +0.0775

### Dialogue

```
#Person1#: Excuse me.
#Person2#: Yeah?
#Person1#: I've never used this place before. Can you give me some idea what I need to do?
#Person2#: You just put quarters in the machines. It's easy.
#Person1#: Yes, but. . . well. . .
#Person2#: What?
#Person1#: How do I use the machines?
#Person2#: What do you mean? These are the washing machines. Those big things over there are the dryers.
#Person1#: I see. Do the machines have soap in them?
#Person2#: No, of course not. You have to put soap in. Did you bring your soap?
#Person1#: No. I don't have soap.
#Person2#: Well, you can buy some from that vending machine over there.
#Person1#: Thanks. Okay. I have my soap.
#Person2#: My God! You really bought a lot. Why do you need so much?
#Person1#: I don't know. I want my clothes to be clean.
#Person2#: But you can't use so much. The machine won't be able to rinse the soap out.
#Person1#: Oh. I guess I didn't know. I have never washed clothes before.
#Person2#: What? Did you say you never washed clothes before?
#Person1#: Yes.
#Person2#: In your life? Are you kidding? Never?
#Person1#: No. Never.
#Person2#: I can't believe it. How can that be? How old are you?
#Person1#: I'm nineteen.
#Person2#
[... truncated for display]
```

### Human references (3)

1. Nick asks Alice how to use washing machines and dryers. Alice teaches him and she's surprised to find that he never washed clothes before. Nick tells her he is from Taiwan, where the mother does everything for the kids. Alice will help Nick to learn how to live on his own.
2. Nick is new here and he has never done washing himself. So he asks a girl called Alice for help. Alice's surprised but still teaches him how to use the machines. Nick tells her his mother used to do everything for him.
3. Nick has never used washing machines before, and he turns to a girl for help. Nick appreciates Alice's help and tells her his mother used to do washing for him. Alice will help him be more independent.

### Base output

> Nick, a nineteen-year-old from Taiwan, is visiting a laundry facility and is unsure how to use the machines. Alice, the host, explains that the machines require soap and that Nick needs to bring his own. Nick is surprised to learn that he has never washed clothes before, and Alice asks him how he plans to survive on his own in a new country. Nick admits he doesn't know how to do it, but he says he has to learn. Alice offers to help

### Fine-tuned output

> Nick asks Alice how to use the washing machines. Alice tells Nick that he needs to buy soap from the vending machine and he needs to use less soap. Alice thinks Nick can't do anything for himself and asks Nick how he's going to survive here. Alice will help Nick learn how to use the machines.

---

## 50% (median) — test row 179

ROUGE-L: base 0.3208 | tuned 0.5000 | Δ +0.1792

### Dialogue

```
#Person1#: Have you applied for you visa to go to study in the united states yet?
#Person2#: Yes, I have. I handed in my application form two weeks ago and I'm going for an interview next week.
#Person1#: Good luck! I'Ve heard it's very difficult to get a visa to go to the united states.
#Person2#: The application form are quite complicated. It took me a few hours to fill it out. The hardest thing was getting all the necessary documents. That took almost two weeks.
#Person1#: What kind of documents did you need to present?
#Person2#: I had to show document relating to my financial status and of course my education, because I want a student visa.
#Person1#: Why do you think so many people have their applications rejected?
#Person2#: I think that they don't complete the forms correctly or they don't include all the required documents. The embassy is very strict about it. You have to be quite careful.
#Person1#: Why are they stricter with Chinese people than with other nationalities?
#Person2#: That's simple. Many people break the rules regarding their visa conditions.
```

### Human references (3)

1. #Person2# has applied for a student visa in the states and tells #Person1# about the rules and the required documents of applying for a USA visa and why they are so strict.
2. #Person2# tells #Person1# #Person2#'s applied for the student visa to study in the united states. They talk about the procedures, materials, and points for attention of the application.
3. #Person2# tells #Person1# that #Person2# has applied for a student visa to the United States. #Person2# says many people are rejected because they are not careful about it.

### Base output

> #Person2# has applied for a student visa to study in the United States. The process was challenging, taking two weeks to complete the application form and almost two weeks to gather all necessary documents. The hardest part was ensuring all required documents were included. #Person1# advises #Person2# to be careful and ensure the application is completed correctly. #Person2# believes the stricter visa conditions for Chinese applicants are due to common violations of visa rules.

### Fine-tuned output

> #Person2# tells #Person1# that #Person2# has applied for a visa to go to study in the United States and #Person2# had to show financial status and education documents. #Person2# thinks many people have their applications rejected because they don't complete the forms correctly or don't include all the required documents.

---

## 75% — test row 38

ROUGE-L: base 0.1463 | tuned 0.4314 | Δ +0.2850

### Dialogue

```
#Person1#: Maggie, can I borrow your notes for history? I'll return them tomorrow.
#Person2#: Sorry, but I usually go to the cafeteria and review them. Why not copy them in the library?
#Person1#: OK.
#Person2#: Here you are.
#Person1#: You are a great help, Maggie.
#Person2#: I don't quite understand a why you need my notes, Mark? You haven't missed any classes.
#Person1#: To be honest, I work in supermarket from 7 to 10 o'clock every evening.
#Person2#: I see. So you're pretty tired when you come to class.
#Person1#: That's exactly why I want to borrow your notes. My notes aren't very good.
#Person2#: So what do you usually do in class?
#Person1#: I'm always awake at the beginning, but thirty minutes after class begins I feel sleepy and have trouble keeping my eyes open.
#Person2#: Well, I need someone to study with and you need someone to keep you awake. Can we be study partners?
#Person1#: Oh sure. That's a good idea. So give me a push when you see me sleeping in class.
#Person2#: OK. Let's start today in the library. We are going there anyway and I don't have to go to the cafeteria.
#Person1#: Sounds good.
```

### Human references (3)

1. Mark wants to borrow Maggie's class notes. Maggie suggests Mark copy them in the library and invites him to be study partners.
2. Mark asks Maggie for her history notes because Mark has been too tired in class. They become study partners at the end.
3. Mark asks Maggie for her notes because Mark's notes aren't good. They decide to be study partners and study together.

### Base output

> #Person1# and #Person2# are discussing their study habits and the possibility of becoming study partners. #Person1# is tired during class due to working in a supermarket from 7 to 10 PM, while #Person2# prefers to review notes in the cafeteria. They agree to study together in the library, with #Person2# taking the initiative to keep #Person1# awake during class.

### Fine-tuned output

> Mark asks Maggie to lend him her notes for history because he works in the supermarket and feels sleepy in class. Maggie agrees and they will be study partners.

---

## 100% (best for tuned) — test row 267

ROUGE-L: base 0.1579 | tuned 1.0000 | Δ +0.8421

### Dialogue

```
#Person1#: Congratulations! Mr. Stuart! You won the city marathon.
#Person2#: Thank you. I am very excited.
#Person1#: We are all very proud of you.
#Person2#: I am very happy, too. It was a big game and I won.
#Person1#: I watched the race with my children. We saw you run to the finish line. We all screamed with excitement.
#Person2#: It was really an exciting moment.
```

### Human references (2)

1. #Person1# congratulates Mr. Stuart on winning a marathon.
2. #Person1# congratulates Mr. Stuart on winning the city marathon.

### Base output

> #Person2# won the city marathon, and #Person1# expressed pride and excitement. #Person1# watched the race with their children, who screamed with joy upon seeing #Person2# cross the finish line.

### Fine-tuned output

> #Person1# congratulates Mr. Stuart on winning the city marathon.

