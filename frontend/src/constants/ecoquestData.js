/**
 * EcoQuest – Extended Question Bank & Educational Curricula
 * 16 Diverse Waste Scenarios covering all 7 categories and real-world contamination traps.
 * Categories: plastic, metal, cardboard, glass, paper, general_trash, organic
 */

export const ECOQUEST_QUESTIONS = [
  // 1. Pizza Box
  {
    id: 'pizza-box',
    title: 'Greasy Delivery Pizza Box',
    image: '/ecoquest/pizza_box.jpg',
    category: 'general_trash',
    categoryLabel: 'General Trash',
    scenario:
      'You finished a hot pizza. The bottom of the cardboard box is heavily soaked with cheese grease and tomato sauce.',
    options: [
      {
        id: 'A',
        text: 'Blue Recyclable Bin (Paper/Cardboard)',
        isCorrect: false,
        feedback: 'Incorrect. Food oil binds to cardboard fibers and ruins recycled paper slurries.',
      },
      {
        id: 'B',
        text: 'General Trash (or tear clean top lid for Cardboard Recycling)',
        isCorrect: true,
        feedback: 'Correct! Tear off the clean top lid to recycle in Blue Bin; put the grease-soaked bottom in General Trash or home compost.',
      },
      {
        id: 'C',
        text: 'Wash the box with warm dish soap and dry in the sun',
        isCorrect: false,
        feedback: 'Incorrect. Cardboard dissolves into mush in water, destroying the fibers.',
      },
      {
        id: 'D',
        text: 'Hazardous Chemical Waste Bin',
        isCorrect: false,
        feedback: 'Incorrect. Food grease is organic food residue, not toxic industrial chemical waste.',
      },
    ],
    explanation:
      'Cardboard recycling relies on clean water-based pulp mixing. Food grease repels water and cannot be filtered out during paper milling.',
    disposalTip:
      '💡 Pro Tip: Tear the box! Clean unsoiled tops go to Cardboard Recycling (Blue Bin); greasy bottoms go to General Trash.',
    localDisclaimer:
      'Check local guidelines: Some municipal compost systems with industrial digesters accept food-soiled cardboard.',
  },

  // 2. Plastic PET Bottle
  {
    id: 'plastic-bottle',
    title: 'Empty PET Beverage Bottle',
    image: '/ecoquest/plastic_bottle.jpg',
    category: 'plastic',
    categoryLabel: 'Plastic',
    scenario:
      'A clear 500ml water bottle (#1 PET) with its screw cap and plastic label attached.',
    options: [
      {
        id: 'A',
        text: 'Discard in General Trash with remaining water inside',
        isCorrect: false,
        feedback: 'Incorrect. PET is highly recyclable and should never be sent to landfills.',
      },
      {
        id: 'B',
        text: 'Empty liquid, crush bottle, replace cap, and place in Blue Recycling Bin',
        isCorrect: true,
        feedback: 'Correct! Rinsing and crushing saves collection space. Modern recycling plants separate caps in water flotation tanks.',
      },
      {
        id: 'C',
        text: 'Burn in an open fireplace or garden pit',
        isCorrect: false,
        feedback: 'Incorrect. Burning plastics releases toxic dioxins and respiratory pollutants.',
      },
      {
        id: 'D',
        text: 'Green Organic Composting Bin',
        isCorrect: false,
        feedback: 'Incorrect. Petroleum-derived PET plastic will not decompose in organic compost.',
      },
    ],
    explanation:
      'Clear PET bottles can be remade into new bottles or polyester textiles infinitely if collected clean and unsoiled.',
    disposalTip:
      '💡 Pro Tip: Leave the cap screwed on after crushing! Small loose caps fall through sorting screen grates.',
    localDisclaimer:
      'Some local regions collect caps separately; verify with your local dry waste drop-off center.',
  },

  // 3. Broken Glass
  {
    id: 'broken-glass',
    title: 'Shattered Beverage Glass Bottle',
    image: '/ecoquest/broken_glass.jpg',
    category: 'glass',
    categoryLabel: 'Glass',
    scenario:
      'A glass bottle shattered on the floor into dozens of sharp, jagged glass shards.',
    options: [
      {
        id: 'A',
        text: 'Throw loose bare shards directly into the dry recycling bin',
        isCorrect: false,
        feedback: 'Incorrect! Loose broken glass cuts sanitation workers and contaminates other dry recyclables.',
      },
      {
        id: 'B',
        text: 'Wrap securely in newspaper/carton, label "SHARP GLASS", and place in safety waste',
        isCorrect: true,
        feedback: 'Correct! Worker safety is paramount. Wrap in thick layers of puncture-proof paper or cardboard before disposal.',
      },
      {
        id: 'C',
        text: 'Bury the glass shards in garden soil',
        isCorrect: false,
        feedback: 'Incorrect. Glass does not decompose and creates severe injury hazards for gardeners and wildlife.',
      },
      {
        id: 'D',
        text: 'Flush small pieces down the kitchen garbage disposal',
        isCorrect: false,
        feedback: 'Incorrect. Heavy glass shards jam disposal blades and clog plumbing bends.',
      },
    ],
    explanation:
      'While container glass is endlessly recyclable, shattered loose shards pose severe physical injury risks to waste handlers.',
    disposalTip:
      '💡 Safety Rule: Municipal Solid Waste guidelines require sharp broken glass to be securely wrapped in 3-4 layers of newspaper.',
    localDisclaimer:
      'Some cities provide dedicated neighborhood glass bottle banks; shattered glass is usually directed to wrapped residual waste.',
  },

  // 4. Fruit and Vegetable Peels
  {
    id: 'fruit-peels',
    title: 'Kitchen Fruit & Vegetable Trimmings',
    image: '/ecoquest/fruit_peels.jpg',
    category: 'organic',
    categoryLabel: 'Organic',
    scenario:
      'A cutting board with citrus peels, banana skins, potato peelings, and vegetable scraps after preparing dinner.',
    options: [
      {
        id: 'A',
        text: 'Seal in a plastic shopping bag and throw in Blue Recyclables',
        isCorrect: false,
        feedback: 'Incorrect. Wet food scraps ruin clean paper and cardboard in recyclables.',
      },
      {
        id: 'B',
        text: 'Green Bin (Wet / Organic Waste) or home compost pile',
        isCorrect: true,
        feedback: 'Correct! Raw organic scraps decompose naturally into nutrient-dense compost for topsoil regeneration.',
      },
      {
        id: 'C',
        text: 'Dry Paper Recycling Bin',
        isCorrect: false,
        feedback: 'Incorrect. Moisture triggers fungal rot on dry paper products.',
      },
      {
        id: 'D',
        text: 'Metal Recycling Caddy',
        isCorrect: false,
        feedback: 'Incorrect. Vegetable matter is biological waste, not metal.',
      },
    ],
    explanation:
      'Composting organic scraps prevents them from rotting anaerobically in landfills, which releases harmful methane gas.',
    disposalTip:
      '💡 Pro Tip: Peel off plastic PLU barcode stickers from fruit skins before composting—they are made of non-biodegradable vinyl!',
    localDisclaimer:
      'Home composters should balance wet peels with dry leaves/brown paper in a 1:2 ratio to maintain aeration.',
  },

  // 5. Paper Tissues
  {
    id: 'paper-tissue',
    title: 'Used Paper Napkins & Facial Tissues',
    image: '/ecoquest/paper_tissue.jpg',
    category: 'general_trash',
    categoryLabel: 'General Trash',
    scenario:
      'Used white paper tissues and napkins used to wipe food spills and hands after eating.',
    options: [
      {
        id: 'A',
        text: 'Blue Bin with high-grade office copy paper',
        isCorrect: false,
        feedback: 'Incorrect. Tissue fibers are too short to bind into recycled paper, plus they carry hygiene contaminants.',
      },
      {
        id: 'B',
        text: 'Black/Grey Bin (General Trash) or approved municipal wet compost',
        isCorrect: true,
        feedback: 'Correct! Used napkins belong in General Trash (or green wet compost if food-only and unbleached).',
      },
      {
        id: 'C',
        text: 'Plastic Film Recycling Bin',
        isCorrect: false,
        feedback: 'Incorrect. Tissues are cellulose wood fibers, not polymer plastic.',
      },
      {
        id: 'D',
        text: 'Metal and Can Bin',
        isCorrect: false,
        feedback: 'Incorrect. Tissues contain zero metallic content.',
      },
    ],
    explanation:
      'Paper napkins and tissues are already at the end of their fiber life cycle. Their cellulose fibers cannot be re-pulped.',
    disposalTip:
      '💡 Hygiene Rule: Tissues used for sneezes or illness should always go into General Trash to safeguard sanitation workers.',
    localDisclaimer:
      'Some municipal wet composters accept unbleached food napkins, but sanitary tissues must always go to General Trash.',
  },

  // 6. Soda Can
  {
    id: 'soda-can',
    title: 'Aluminium Beverage Can',
    image: '/ecoquest/soda_can.jpg',
    category: 'metal',
    categoryLabel: 'Metal',
    scenario:
      'An empty cold aluminium beverage can with its pull-tab still attached.',
    options: [
      {
        id: 'A',
        text: 'Discard in landfill trash',
        isCorrect: false,
        feedback: 'Incorrect. Aluminium is 100% recyclable infinitely with zero degradation.',
      },
      {
        id: 'B',
        text: 'Rinse liquid out and place in Blue Recycling Bin',
        isCorrect: true,
        feedback: 'Correct! Recycling aluminium saves 95% of the massive electricity needed to produce virgin aluminium.',
      },
      {
        id: 'C',
        text: 'Organic Garden Composting',
        isCorrect: false,
        feedback: 'Incorrect. Aluminium is an elemental metal and will never biodegrade.',
      },
      {
        id: 'D',
        text: 'Rip the pull-tab off and toss it on the sidewalk',
        isCorrect: false,
        feedback: 'Incorrect. Detached pull-tabs pose choking hazards to urban birds and wildlife.',
      },
    ],
    explanation:
      'Recycling just 1 aluminium can saves enough electricity to power a laptop for 3 hours. A recycled can returns to store shelves in 60 days.',
    disposalTip:
      '💡 Pro Tip: Keep the pull-tab pushed flat against the can lid so the entire aluminium piece stays intact during optical sorting.',
    localDisclaimer:
      'In regions with container deposit legislation (DRS), return cans to retail reverse vending machines for cash deposits.',
  },

  // 7. Clean Cardboard Packaging
  {
    id: 'cardboard-box',
    title: 'Clean Corrugated Shipping Carton',
    image: '/ecoquest/cardboard_box.jpg',
    category: 'cardboard',
    categoryLabel: 'Cardboard',
    scenario:
      'A dry, clean brown e-commerce shipping carton with paper tape and no food spills.',
    options: [
      {
        id: 'A',
        text: 'Flatten completely and place in Blue Recycling Bin',
        isCorrect: true,
        feedback: 'Correct! Flattening boxes increases truck collection efficiency by 400% and prevents hopper jams.',
      },
      {
        id: 'B',
        text: 'Leave box open in the rain outside before pickup',
        isCorrect: false,
        feedback: 'Incorrect. Wet cardboard attracts mildew and weakens cellulose fiber strength.',
      },
      {
        id: 'C',
        text: 'General Trash Landfill Bin',
        isCorrect: false,
        feedback: 'Incorrect. Clean corrugated cardboard has high commercial recycling value.',
      },
      {
        id: 'D',
        text: 'Hazardous Waste Drum',
        isCorrect: false,
        feedback: 'Incorrect. Shipping cartons are non-hazardous packaging materials.',
      },
    ],
    explanation:
      'Corrugated shipping boxes are made from long, strong pine fibers that can be re-pulped into new boxes up to 5-7 times.',
    disposalTip:
      '💡 Space Saver: Slice box tape along seams and flatten flat to maximize space in your recycling caddy.',
    localDisclaimer:
      'Remove bulky styrofoam or plastic bubble inserts before placing the cardboard box into recycling bins.',
  },

  // 8. Chip Packet (Multilayer Plastic)
  {
    id: 'mixed-waste',
    title: 'Multilayer Chip & Snack Packet',
    image: '/ecoquest/mixed_waste.jpg',
    category: 'general_trash',
    categoryLabel: 'General Trash',
    scenario:
      'A shiny potato chip packet made of metallized plastic film (BOPP plastic outer with vacuum aluminium vapor barrier).',
    options: [
      {
        id: 'A',
        text: 'Blue Bin with pure aluminium soda cans',
        isCorrect: false,
        feedback: 'Incorrect! The interior shine is microscopically thin aluminium vapor fused to plastic; it cannot be melted as metal.',
      },
      {
        id: 'B',
        text: 'Black/Grey Bin (General Trash / Refuse Derived Fuel)',
        isCorrect: true,
        feedback: 'Correct! Multilayer packaging (MLP) cannot be separated into pure resin streams with standard mechanical recycling.',
      },
      {
        id: 'C',
        text: 'Green Bin for biological garden composting',
        isCorrect: false,
        feedback: 'Incorrect. Plastic films do not biodegrade and will contaminate soil with microplastics.',
      },
      {
        id: 'D',
        text: 'Paper Bin with newspapers',
        isCorrect: false,
        feedback: 'Incorrect. Chip packets contain zero paper or cellulose pulp.',
      },
    ],
    explanation:
      'Multilayer plastics (MLP) fuse different polymers together for freshness. Standard mechanical recyclers cannot separate the layers.',
    disposalTip:
      '💡 Simple Test: Do the "Crinkle & Tear" test! If it has a shiny metallic interior but stretches like plastic, it is non-recyclable MLP.',
    localDisclaimer:
      'In India, CPCB Extended Producer Responsibility (EPR) mandates collecting MLP for co-processing in cement kilns (Refuse Derived Fuel).',
  },

  // 9. Disposable Coffee Cup
  {
    id: 'coffee-cup',
    title: 'Takeaway Disposable Coffee Cup',
    image: '/ecoquest/coffee_cup.jpg',
    category: 'general_trash',
    categoryLabel: 'General Trash',
    scenario:
      'A paper coffee cup from a cafe. The outside is paperboard, but the inside has a waterproof polyethylene plastic lining to prevent leaks.',
    options: [
      {
        id: 'A',
        text: 'Standard Paper Recycling Bin',
        isCorrect: false,
        feedback: 'Incorrect! The plastic polyethylene lining clogs traditional paper mill hydro-pulpers, ruining the batch.',
      },
      {
        id: 'B',
        text: 'General Trash (unless your city has a dedicated cup recycling facility)',
        isCorrect: true,
        feedback: 'Correct! Over 99% of paper coffee cups end up in General Trash because separating the inner plastic layer requires specialized chemical recycling.',
      },
      {
        id: 'C',
        text: 'Metal Recycling Caddy',
        isCorrect: false,
        feedback: 'Incorrect. Coffee cups are lined paperboard, not metal.',
      },
      {
        id: 'D',
        text: 'Organic Garden Compost',
        isCorrect: false,
        feedback: 'Incorrect. Standard coffee cups are lined with petroleum plastic that will never decompose in garden soil.',
      },
    ],
    explanation:
      'To hold hot liquid without leaking, paper cups are bonded with a thin plastic film. Standard paper recycling mills cannot separate this film.',
    disposalTip:
      '💡 Eco Switch: Bring your own reusable stainless steel travel mug! Most cafes offer discounts for BYO tumblers.',
    localDisclaimer:
      'Plastic lid #5 (PP) can often be recycled in Blue Bins; check the number molded on the cup lid.',
  },

  // 10. Milk Carton / Tetra Pak
  {
    id: 'milk-carton',
    title: 'Aseptic Milk Carton (Tetra Pak)',
    image: '/ecoquest/milk_carton.jpg',
    category: 'cardboard',
    categoryLabel: 'Cardboard',
    scenario:
      'An empty shelf-stable milk or juice carton with paperboard, plastic barrier, and microscopic aluminium foil layers.',
    options: [
      {
        id: 'A',
        text: 'Rinse clean, flatten, and place in Blue Recycling Bin / Carton Drop-off',
        isCorrect: true,
        feedback: 'Correct! Specialized carton recycling plants use high-consistency pulpers to extract high-value virgin paper fibers and PolyAl.',
      },
      {
        id: 'B',
        text: 'Home vegetable compost pile',
        isCorrect: false,
        feedback: 'Incorrect. The plastic and aluminium layers will not degrade in backyard compost.',
      },
      {
        id: 'C',
        text: 'Toss directly into garden soil',
        isCorrect: false,
        feedback: 'Incorrect. Plastics inside the carton persist for hundreds of years.',
      },
      {
        id: 'D',
        text: 'Burn in an open fire',
        isCorrect: false,
        feedback: 'Incorrect. Burning composite cartons releases harmful fumes and greenhouse gases.',
      },
    ],
    explanation:
      'Aseptic cartons consist of 75% paperboard, 20% polyethylene, and 5% aluminium. Specialized hydro-pulpers recover 100% of the paper fibers.',
    disposalTip:
      '💡 Pro Tip: Rinse thoroughly and flatten! Do not crush into a ball—flat cartons are easiest for optical sorters to detect.',
    localDisclaimer:
      'Check local municipal guidelines: Some cities collect cartons in paper streams; others designate them with mixed containers.',
  },

  // 11. Aluminium Food Foil
  {
    id: 'aluminum-foil',
    title: 'Used Food Wrapping Aluminium Foil',
    image: '/ecoquest/aluminum_foil.jpg',
    category: 'metal',
    categoryLabel: 'Metal',
    scenario:
      'Aluminium foil used to wrap a sandwich or roti. There are light food crumbs on it, but no heavy liquid grease.',
    options: [
      {
        id: 'A',
        text: 'Wipe/rinse off food crumbs, roll into a fist-sized ball, and place in Blue Recycling Bin',
        isCorrect: true,
        feedback: 'Correct! Pure aluminium foil is 100% recyclable. Balling it up prevents small thin scraps from blowing away or falling through sorting grates.',
      },
      {
        id: 'B',
        text: 'Shred into tiny confetti and scatter in garden',
        isCorrect: false,
        feedback: 'Incorrect. Metal foil will not degrade in soil and creates litter.',
      },
      {
        id: 'C',
        text: 'Organic Food Compost Bin',
        isCorrect: false,
        feedback: 'Incorrect. Aluminium is inorganic metal, not biodegradable biomass.',
      },
      {
        id: 'D',
        text: 'Paper Recycling Bin',
        isCorrect: false,
        feedback: 'Incorrect. Foil is elemental aluminium, not wood pulp.',
      },
    ],
    explanation:
      'Like aluminium cans, clean foil can be melted down endlessly. If heavily caked with burnt grease, it should go to General Trash.',
    disposalTip:
      '💡 The Ball Rule: Collect small pieces of clean foil until you can ball them together into a 2-3 inch sphere before placing in Blue Bins!',
    localDisclaimer:
      'If aluminium foil is heavily soiled with meat grease or baked-on cheese, place it in General Trash.',
  },

  // 12. Glass Wine Bottle
  {
    id: 'wine-bottle',
    title: 'Empty Glass Wine or Olive Oil Bottle',
    image: '/ecoquest/wine_bottle.jpg',
    category: 'glass',
    categoryLabel: 'Glass',
    scenario:
      'An intact green glass wine or olive oil bottle with a natural cork or metal cap removed.',
    options: [
      {
        id: 'A',
        text: 'Rinse thoroughly and place in Glass Bank or Blue Recycling Bin',
        isCorrect: true,
        feedback: 'Correct! Intact container glass can be melted and remolded endlessly with zero quality loss and lower furnace temperatures.',
      },
      {
        id: 'B',
        text: 'Smash with a hammer before throwing into the recycling bin',
        isCorrect: false,
        feedback: 'Incorrect! Never smash glass bottles—it creates dangerous sharps and prevents color-sorting at cullet processing plants.',
      },
      {
        id: 'C',
        text: 'General Trash Landfill Bin',
        isCorrect: false,
        feedback: 'Incorrect. Glass takes 1 million years to decompose in a landfill; always recycle intact glass containers.',
      },
      {
        id: 'D',
        text: 'Green Organic Compost',
        isCorrect: false,
        feedback: 'Incorrect. Glass is made from silica sand and will never compost.',
      },
    ],
    explanation:
      'Recycled crushed glass (cullet) melts at a much lower temperature than raw silica sand, significantly reducing carbon emissions during glass manufacturing.',
    disposalTip:
      '💡 Color Sorting: Many cities maintain color-separated glass bins (Clear, Green, Amber). Separate colors when possible for higher cullet value.',
    localDisclaimer:
      'Remove metal screw caps (recycle with metals) and corks (compost or reuse) before recycling glass bottles.',
  },

  // 13. Printed Newspapers
  {
    id: 'newspaper',
    title: 'Read Daily Printed Newspapers',
    image: '/ecoquest/newspaper.jpg',
    category: 'paper',
    categoryLabel: 'Paper',
    scenario:
      'A bundle of clean, dry printed newspapers and weekly magazines after reading.',
    options: [
      {
        id: 'A',
        text: 'Keep clean and dry, bundle, and place in Blue Recycling Bin or give to local paper vendor',
        isCorrect: true,
        feedback: 'Correct! Dry newsprint is easily de-inked and recycled into egg cartons, tissue paper, or new newsprint.',
      },
      {
        id: 'B',
        text: 'Soak with water and mix with wet vegetable peels',
        isCorrect: false,
        feedback: 'Incorrect. High moisture ruins paper quality for mechanical paper recycling.',
      },
      {
        id: 'C',
        text: 'General Trash Landfill Bin',
        isCorrect: false,
        feedback: 'Incorrect. Clean paper should always be recycled rather than landfilled.',
      },
      {
        id: 'D',
        text: 'Plastic Bin',
        isCorrect: false,
        feedback: 'Incorrect. Newspapers are 100% cellulose fiber, not synthetic polymers.',
      },
    ],
    explanation:
      'Recycling 1 ton of newspaper saves 17 mature trees, 7,000 gallons of water, and 4,000 kWh of electricity.',
    disposalTip:
      '💡 Keep it Dry: Moisture weakens paper fibers and triggers mildew. Always store paper indoors until collection day.',
    localDisclaimer:
      'In India, local kabadiwalas (informal waste pickers) buy old newspapers (raddi) at fair market rates for domestic paper mills.',
  },

  // 14. Styrofoam Packaging
  {
    id: 'styrofoam',
    title: 'Expanded Polystyrene (Styrofoam) Block',
    image: '/ecoquest/styrofoam.jpg',
    category: 'general_trash',
    categoryLabel: 'General Trash',
    scenario:
      'A large white molded polystyrene foam block from an appliance shipping box (#6 EPS).',
    options: [
      {
        id: 'A',
        text: 'Curbside Blue Paper Recycling Bin',
        isCorrect: false,
        feedback: 'Incorrect. Polystyrene is expanded plastic, not paperboard.',
      },
      {
        id: 'B',
        text: 'General Trash (or dedicated EPS drop-off center if available)',
        isCorrect: true,
        feedback: 'Correct! Styrofoam is 95% air; standard municipal sorting facilities do not accept it in curbside recycling bins.',
      },
      {
        id: 'C',
        text: 'Organic Garden Composting',
        isCorrect: false,
        feedback: 'Incorrect. Styrofoam takes 500+ years to break down and fragments into hazardous microplastics.',
      },
      {
        id: 'D',
        text: 'Burn in an open yard',
        isCorrect: false,
        feedback: 'Incorrect. Burning polystyrene releases styrene gas and dense toxic black smoke.',
      },
    ],
    explanation:
      'EPS is extremely lightweight and fragile. During standard sorting, it fractures into millions of tiny static beads that contaminate all other recyclables.',
    disposalTip:
      '💡 Tip: Some electronics retailers take back molded EPS blocks for specialized densification and molding into picture frames.',
    localDisclaimer:
      'Check local drop-offs: Some specialized municipal depots have foam densifiers; never place loose foam in standard blue bins.',
  },

  // 15. Spent Coffee Grounds
  {
    id: 'coffee-grounds',
    title: 'Used Espresso Coffee Grounds',
    image: '/ecoquest/coffee_grounds.jpg',
    category: 'organic',
    categoryLabel: 'Organic',
    scenario:
      'Dark moist coffee grounds left in the portafilter or coffee filter paper after brewing morning coffee.',
    options: [
      {
        id: 'A',
        text: 'Green Bin (Wet / Organic Waste) or mix into garden soil/compost',
        isCorrect: true,
        feedback: 'Correct! Coffee grounds are rich in nitrogen, phosphorus, and potassium, making them fantastic fertilizer and compost green material.',
      },
      {
        id: 'B',
        text: 'Blue Recyclables Bin with plastic and metal cans',
        isCorrect: false,
        feedback: 'Incorrect. Wet grounds will splatter and contaminate dry recyclables.',
      },
      {
        id: 'C',
        text: 'Dump into sink drain pipes',
        isCorrect: false,
        feedback: 'Incorrect. Heavy coffee grounds clump inside grease traps and cause severe plumbing clogs.',
      },
      {
        id: 'D',
        text: 'Dry Paper Bin',
        isCorrect: false,
        feedback: 'Incorrect. Moisture will destroy the dry paper pulp.',
      },
    ],
    explanation:
      'Used coffee grounds contain roughly 2% nitrogen by volume and attract earthworms in garden compost, accelerating soil regeneration.',
    disposalTip:
      '💡 Garden Hack: Sprinkle coffee grounds around acid-loving plants like roses, tomatoes, and hydrangeas, or mix with home compost!',
    localDisclaimer:
      'Unbleached paper coffee filters can be composted along with the grounds; remove plastic K-cups first.',
  },

  // 16. Single-Use Thin Plastic Grocery Bag
  {
    id: 'plastic-bag',
    title: 'Thin Low-Density Polyethylene Shopping Bag',
    image: '/ecoquest/plastic_bag.jpg',
    category: 'general_trash',
    categoryLabel: 'General Trash',
    scenario:
      'A thin crinkly plastic shopping bag (#4 LDPE). It is clean and dry, but very stretchy and lightweight.',
    options: [
      {
        id: 'A',
        text: 'Toss loose into the curbside Blue Recycling Bin',
        isCorrect: false,
        feedback: 'Incorrect! Thin plastic films are known as "tanglers"—they wrap around rotating sorting disks at MRFs and cause expensive shutdowns.',
      },
      {
        id: 'B',
        text: 'General Trash (or dedicated retail soft-plastic drop-off kiosk)',
        isCorrect: true,
        feedback: 'Correct! Plastic film requires dedicated drop-off bins at supermarkets. Loose bags in curbside bins destroy sorting machinery.',
      },
      {
        id: 'C',
        text: 'Green Food Compost Bin',
        isCorrect: false,
        feedback: 'Incorrect. Standard grocery bags will never biodegrade in organic compost.',
      },
      {
        id: 'D',
        text: 'Cardboard Bin',
        isCorrect: false,
        feedback: 'Incorrect. Thin plastic film is not cardboard.',
      },
    ],
    explanation:
      'Plastic film jams the mechanical sorting machinery at recycling plants, requiring operators to stop equipment and cut bags out by hand.',
    disposalTip:
      '💡 Retail Drop-Off: Many major grocery supermarkets offer dedicated "Soft Plastics Drop-Off" bins at front entrances.',
    localDisclaimer:
      'Under India Plastic Waste Management Rules, carry bags under 120 microns are banned; switch to reusable cotton cloth bags.',
  },
];
